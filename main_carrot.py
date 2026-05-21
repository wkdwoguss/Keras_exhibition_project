import cv2
import torch
import torch.nn as nn
import os
import numpy as np
from torchvision import models, transforms
from PIL import Image
from ultralytics import YOLO
from rembg import remove, new_session

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

yolo = YOLO('yolov8x.pt')
CARROT_CLASS_ID = 51

rembg_session = new_session(providers=['CUDAExecutionProvider'])

def remove_green(img):
    img_array = np.array(img)
    hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
    lower_green = np.array([30, 35, 35])
    upper_green = np.array([85, 255, 255])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    kernel = np.ones((45, 45), np.uint8)
    dilated_mask = cv2.dilate(green_mask, kernel, iterations=2)
    lower_white = np.array([0, 0, 180])
    upper_white = np.array([180, 60, 255])
    white_mask = cv2.inRange(hsv, lower_white, upper_white)
    nearby_white = cv2.bitwise_and(dilated_mask, white_mask)
    final_mask = cv2.bitwise_or(green_mask, nearby_white)
    img_array[final_mask > 0] = [0, 0, 0]
    return Image.fromarray(img_array)


def crop_carrot(img):
    img_array = np.array(img)
    mask = img_array.sum(axis=2) > 30
    if mask.sum() < img_array.shape[0] * img_array.shape[1] * 0.05:
        return img
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    pad = 20
    rmin = max(0, rmin - pad)
    rmax = min(img_array.shape[0], rmax + pad)
    cmin = max(0, cmin - pad)
    cmax = min(img_array.shape[1], cmax + pad)
    return img.crop((cmin, rmin, cmax, rmax))


val_transforms = transforms.Compose([
    transforms.Lambda(remove_green),
    transforms.Lambda(crop_carrot),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def load_model(weights_path):
    model = models.efficientnet_v2_s(weights=None)
    num_ftrs = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(num_ftrs, 2)
    )
    state_dict = torch.load(weights_path, map_location=device)
    if any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model

model1 = load_model('first_algo/carrot_weights_1.pth')
model2 = load_model('first_algo/carrot_weights_2.pth')

label_colors = {
    'Good':      (0, 255, 0),
    'Bad':       (0, 0, 255),
    'Imperfect': (0, 165, 255),
}

def classify(cropped_bgr):
    img_rgb = cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2RGB)
    
    result_rgba = remove(img_rgb, session=rembg_session)
    img_pil_rgba = Image.fromarray(result_rgba)
    img_pil = img_pil_rgba.convert('RGB')
    
    img_arr = np.array(img_pil)
    nonblack_ratio = (img_arr.sum(axis=2) > 30).sum() / (img_arr.shape[0] * img_arr.shape[1])
    
    if nonblack_ratio < 0.05:
        img_pil = Image.fromarray(img_rgb)
    
    img_tensor = val_transforms(img_pil).unsqueeze(0).to(device)
    
    with torch.no_grad():
        out1 = model1(img_tensor)
        pred1 = torch.argmax(out1, dim=1).item()
        if pred1 == 0:
            return 'Bad'
        
        out2 = model2(img_tensor)
        pred2 = torch.argmax(out2, dim=1).item()
        return 'Good' if pred2 == 0 else 'Imperfect'


cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cam_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
cam_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Camera Resolution: {cam_w} x {cam_h}")

cv2.namedWindow("YOLO Detection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("YOLO Detection", cam_w, cam_h)

center_ratio = 0.7
detected_crops = []
detected_ids = set()
id_to_label = {}

BOX_PADDING = 50

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    process_frame = frame.copy()
    display_frame = frame

    h, w, _ = frame.shape
    cx_min = int(w * (0.5 - center_ratio / 2))
    cx_max = int(w * (0.5 + center_ratio / 2))
    cy_min = int(h * (0.5 - center_ratio / 2))
    cy_max = int(h * (0.5 + center_ratio / 2))

    results = yolo.track(process_frame, persist=True, conf=0.1,
                         classes=[CARROT_CLASS_ID],
                         tracker="bytetrack.yaml", verbose=False)
    
    cv2.rectangle(display_frame, (cx_min, cy_min), (cx_max, cy_max), (255, 0, 0), 2)

    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id.cpu().numpy()

        for box, track_id in zip(boxes, track_ids):
            x1, y1, x2, y2 = map(int, box)
            obj_id = int(track_id)

            in_center = (cx_min <= x1 and x2 <= cx_max) and (cy_min <= y1 and y2 <= cy_max)
            if not in_center:
                continue

            if obj_id not in detected_ids:
                crop_y1 = max(0, y1 - BOX_PADDING)
                crop_y2 = min(h, y2 + BOX_PADDING)
                crop_x1 = max(0, x1 - BOX_PADDING)
                crop_x2 = min(w, x2 + BOX_PADDING)
                
                cropped_obj = process_frame[crop_y1:crop_y2, crop_x1:crop_x2]

                if cropped_obj.size > 0:
                    detected_crops.append(cropped_obj)
                    detected_ids.add(obj_id)
                    label = classify(cropped_obj)
                    id_to_label[obj_id] = label
                    print(f"ID: {obj_id} | Classification Result: {label} | Total: {len(detected_crops)}")

            label = id_to_label.get(obj_id, "?")
            color = label_colors.get(label, (200, 200, 200))

            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 3)
            text = f"ID:{obj_id} {label}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            thickness = 2
            (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
            cv2.rectangle(display_frame, (x1, y1 - th - 10), (x1 + tw + 10, y1), color, -1)
            cv2.putText(display_frame, text, (x1 + 5, y1 - 5),
                        font, font_scale, (255, 255, 255), thickness)

    cv2.imshow("YOLO Detection", display_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()