# file of algorithm: carrot
# you can find the details in the research note

import cv2
import numpy as np
from rembg import remove
from pathlib import Path
import glob
import os
import onnxruntime as ort

# Force GPU usage for ONNX Runtime
ort.set_default_logger_severity(3)  # Suppress warnings

def to_rgba(img_bgr):
    """Coverting image color to RGBA to remove the background"""
    if img_bgr.shape[2] == 3:
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGBA)
    return img_bgr

def resize_into_244(canvas_size, rgba):
    """Resizing the image"""

    H, W = rgba.shape[:2]
    target = canvas_size

    # scaling the image (sustaining the ratio)
    scale = min(target/W, target/H)
    new_w = max(1, int(round(W * scale)))
    new_h = max(1, int(round(H * scale)))
    obj = cv2.resize(rgba, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    # moving image to the center
    canvas = np.zeros((target, target, 4), dtype=np.uint8)
    x0 = (target - new_w) // 2
    y0 = (target - new_h) // 2
    canvas[y0:y0+new_h, x0:x0+new_w] = obj
    return canvas

def process(path, margin_px=4):
    img_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    rgba = remove(img)

    if isinstance(rgba, bytes):
        rgba = np.frombuffer(rgba, np.uint8)
        rgba = cv2.imdecode(rgba, cv2.IMREAD_UNCHANGED)

    alpha = rgba[:, :, 3]
    ys, xs = np.where(alpha > 0)
    if len(xs) == 0 or len(ys) == 0:
        result = resize_into_244(300, rgba)
    else:
        x1, x2 = max(xs.min() - margin_px, 0), min(xs.max() + margin_px, rgba.shape[1] - 1)
        y1, y2 = max(ys.min() - margin_px, 0), min(ys.max() + margin_px, rgba.shape[0] - 1)
        crop = rgba[y1:y2+1, x1:x2+1]

        a = crop[:, :, 3].copy()
        kernel = np.ones((3, 3), np.uint8)
        a = cv2.morphologyEx(a, cv2.MORPH_CLOSE, kernel, iterations=1)
        crop = crop.copy()
        crop[:, :, 3] = a
        result = resize_into_244(300, crop)
        result = cv2.cvtColor(result, cv2.COLOR_RGBA2BGRA)
    return result


# goodli = []
# badli = []
# goodpath = r'D:\Programming\Python\Keras\algoimage\GOOD'
# badpath = r'D:\Programming\Python\Keras\algoimage\BAD'
# patterns = ('*.jpg', '*.jpeg', '*.png')
# gimpath = []
# bimpath = []
# for ext in patterns:
#     goodli += glob.glob(os.path.join(goodpath, ext))
# for ext in patterns:
#     badli += glob.glob(os.path.join(badpath, ext))

# goodli = sorted(goodli)
# badli = sorted(badli)

# gi = len(goodli)
# bi = len(badli)




def curvature(path):
    """Calculating the contour curvature of picture
    input : path"""
    img = process(path)
    alpha = img[:, :, 3]

    mask = (alpha > 0).astype(np.uint8) * 255
    contour, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contour:
        return None
    
    cnt = max(contour, key=cv2.contourArea)
    pts = cnt[:, 0, :].astype(np.float32)
    n = len(pts)
    curvature = 0

    for i in range(2, n - 2):
        x_2 = pts[i - 2, 0]
        x_1 = pts[i - 1, 0]
        x0 = pts[i, 0]
        x1 = pts[i + 1, 0]
        x2 = pts[i + 2, 0]

        y_2 = pts[i - 2, 1]
        y_1 = pts[i - 1, 1]
        y0 = pts[i, 1]
        y1 = pts[i + 1, 1]
        y2 = pts[i + 2, 1]

        x_prime = (x1 - x_1) / 2.0
        y_prime = (y1 - y_1) / 2.0
        x_pprime = (x2 - 2 * x0 + x_2) / 4.0
        y_pprime = (y2 - 2 * y0 + y_2) / 4.0

        num = abs(x_prime * y_pprime - y_prime * x_pprime)
        den = (x_prime ** 2 + y_prime ** 2) ** 1.5

        if den > 1e-8:
            curvature += num / den
        else:
            curvature += 0

    return curvature


# gcurva = np.float32(0)

# gcnt = 0
# bcnt = 0

# for i in range(gi):
#     temp = curvature(goodli[i])
#     gcurva += temp
#     print(f"Good carrot successfully calculated : {goodli[i]}")
#     gcnt += 1
#     print(f"Good carrot number : {gcnt}, Leftover : {gi - gcnt}")
# gcurva /= np.float32(gi)

# bcurva = np.float32(0)

# for i in range(bi):
#     temp = curvature(badli[i])
#     bcurva += temp
#     print(f"Bad carrot successfully calculated : {badli[i]}")
#     bcnt += 1
#     print(f"Bad carrot number : {bcnt}, Leftover : {bi - bcnt}")
# bcurva /= np.float32(bi)

# print(f"Good carrot average curvature: {gcurva}\nBad carrot average curvature: {bcurva}")
# good carrot average curvature: 145.19886779785156
# bad carrot average curvature: 100.78387451171875

gcurva = 145.19886779785156
bcurva = 100.78387451171875

def sortCarrot(curv):
    if curv >= gcurva:
        return 'good'
    elif curv < gcurva and curv >= bcurva:
        return 'imperfect'
    else:
        return 'bad'

def result(path):
    curv = curvature(path)
    res = sortCarrot(curv)
    return res

