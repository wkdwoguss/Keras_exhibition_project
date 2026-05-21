import cv2
import torch
import torch.nn as nn
import numpy as np
from torchvision import models, transforms
from PIL import Image
from ultralytics import YOLO
from rembg import remove, new_session

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

yolo = YOLO('first_algo/apple_best.pt')
rembg_session = new_session(providers=['CUDAExecutionProvider'])

classifier = models.efficientnet_v2_s(weights=None)
num_ftrs = classifier.classifier[1].in_features
classifier.classifier[1] = nn.Linear(num_ftrs, 3)

state_dict = torch.load('first_algo/apple_weights.pth', map_location=device)
if any(k.startswith('module.') for k in state_dict.keys()):
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
classifier.load_state_dict(state_dict)
classifier = classifier.to(device)
classifier.eval()

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

class_names = ['Bad', 'Good', 'Imperfect']
label_colors = {
    'Good':      (0, 255, 0),
    'Bad':       (0, 0, 255),
    'Imperfect': (0, 165, 255),
}


def is_sharp(image_bgr, threshold):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return variance > threshold


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
        output = classifier(img_tensor)
        pred_idx = torch.argmax(output, dim=1).item()
    
    return class_names[pred_idx]


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
STABLE_FRAMES = 5
MOVE_THRESHOLD = 20
BLUR_THRESHOLD = 100

stable_counter = {}
prev_positions = {}

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

    results = yolo.track(process_frame, persist=True, conf=0.5,
                         tracker="bytetrack.yaml", verbose=False)
    
    cv2.rectangle(display_frame, (cx_min, cy_min), (cx_max, cy_max), (255, 0, 0), 2)

    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id.cpu().numpy()

        for box, track_id in zip(boxes, track_ids):
            x1, y1, x2, y2 = map(int, box)
            obj_id = int(track_id)
            obj_cx = (x1 + x2) // 2
            obj_cy = (y1 + y2) // 2

            if obj_id in prev_positions:
                px, py = prev_positions[obj_id]
                moved = abs(obj_cx - px) + abs(obj_cy - py)
                if moved < MOVE_THRESHOLD:
                    stable_counter[obj_id] = stable_counter.get(obj_id, 0) + 1
                else:
                    stable_counter[obj_id] = 0
            else:
                stable_counter[obj_id] = 0
            prev_positions[obj_id] = (obj_cx, obj_cy)

            in_center = (cx_min <= x1 and x2 <= cx_max) and (cy_min <= y1 and y2 <= cy_max)
            if not in_center:
                continue

            if obj_id not in detected_ids and stable_counter.get(obj_id, 0) >= STABLE_FRAMES:
                crop_y1 = max(0, y1 - BOX_PADDING)
                crop_y2 = min(h, y2 + BOX_PADDING)
                crop_x1 = max(0, x1 - BOX_PADDING)
                crop_x2 = min(w, x2 + BOX_PADDING)
                
                cropped_obj = process_frame[crop_y1:crop_y2, crop_x1:crop_x2]

                if cropped_obj.size > 0:
                    if not is_sharp(cropped_obj, BLUR_THRESHOLD):
                        continue
                    
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
