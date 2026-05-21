import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from ultralytics import YOLO

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

yolo = YOLO('first_algo/apple_best.pt')

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

def classify(cropped_bgr):
    img_rgb = cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    img_tensor = val_transforms(img_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        output = classifier(img_tensor)
        pred_idx = torch.argmax(output, dim=1).item()
    return class_names[pred_idx]


cap = cv2.VideoCapture(0)

cam_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
cam_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Camera Resolution: {cam_w} x {cam_h}")

cv2.namedWindow("YOLO Detection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("YOLO Detection", cam_w, cam_h)

center_ratio = 0.7
detected_crops = []
detected_ids = set()
id_to_label = {}

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    h, w, _ = frame.shape
    cx_min = int(w * (0.5 - center_ratio / 2))
    cx_max = int(w * (0.5 + center_ratio / 2))
    cy_min = int(h * (0.5 - center_ratio / 2))
    cy_max = int(h * (0.5 + center_ratio / 2))

    results = yolo.track(frame, persist=True, conf=0.7,
                         tracker="bytetrack.yaml", verbose=False)
    cv2.rectangle(frame, (cx_min, cy_min), (cx_max, cy_max), (255, 0, 0), 2)

    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id.cpu().numpy()

        for box, track_id in zip(boxes, track_ids):
            x1, y1, x2, y2 = map(int, box)
            obj_id = int(track_id)
            obj_cx = (x1 + x2) // 2
            obj_cy = (y1 + y2) // 2

            in_center = (cx_min <= obj_cx <= cx_max) and (cy_min <= obj_cy <= cy_max)

            if not in_center:
                continue

            if obj_id not in detected_ids:
                crop_y1, crop_y2 = max(0, y1), min(h, y2)
                crop_x1, crop_x2 = max(0, x1), min(w, x2)
                cropped_obj = frame[crop_y1:crop_y2, crop_x1:crop_x2]

                if cropped_obj.size > 0:
                    detected_crops.append(cropped_obj)
                    detected_ids.add(obj_id)
                    label = classify(cropped_obj)
                    id_to_label[obj_id] = label
                    print(f"ID: {obj_id} | Classification Result: {label} | Total: {len(detected_crops)}")

            label = id_to_label.get(obj_id, "?")
            color = label_colors.get(label, (200, 200, 200))

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

            text = f"ID:{obj_id} {label}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            thickness = 2
            (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
            cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 10, y1), color, -1)
            cv2.putText(frame, text, (x1 + 5, y1 - 5),
                        font, font_scale, (255, 255, 255), thickness)

    cv2.imshow("YOLO Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()