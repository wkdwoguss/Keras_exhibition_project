# file of algorithm: apple
# you can find the details in the research note

import cv2
import numpy as np
from second_algo.algocarrot import curvature, process
import glob
import os


# need to know the range of good pixel

# first open the folder !

patterns = ('*.jpg', '*.jpeg', '*.png')
goodli = []
goodpath = r'D:\Programming\Python\Keras\Dataset\Train\Apple\Good'
for ext in patterns:
    goodli += glob.glob(os.path.join(goodpath, ext))

goodli = sorted(goodli)

badli = []
badpath = r'D:\Programming\Python\Keras\Dataset\Train\Apple\Bad'
for ext in patterns:
    badli += glob.glob(os.path.join(badpath, ext))

badli = sorted(badli)

# calculating the averge of hue
# cannot calculate by arithmetic mean

def hue(h_means):
    """calculating average h"""
    rad = np.deg2rad(np.array(h_means) * 2)
    xmean = np.mean(np.cos(rad))
    ymean = np.mean(np.sin(rad))
    theta = np.arctan2(ymean, xmean)
    H = np.rad2deg(theta) / 2
    if H < 0:
        H += 180
    return H
    

def divide(h_means):
    """dividing h array to upper group and lower group"""
    leng = len(h_means)
    left = []
    right = []
    for i in range(leng):
        if h_means[i] < 90:
            left.append(h_means[i])
        else:
            right.append(h_means[i])

    return np.array(left), np.array(right)

def mix(a, b, m=1, n=2):
    rad1 = np.deg2rad(a * 2)
    rad2 = np.deg2rad(b * 2)
    x = (n * np.cos(rad1) + m * np.cos(rad2)) / (m + n)
    y = (n * np.sin(rad1) + m * np.sin(rad2)) / (m + n)
    
    theta = np.arctan2(y, x)
    H = np.rad2deg(theta) / 2
    if H < 0:
        H += 180
    return H
    

def hsvcal(path):
    img = process(path)
    alpha = img[:, :, 3]
    mask = alpha > 0
    bgr = img[:, :, :3]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h = hsv[:, :, 0][mask]; s = hsv[:, :, 1][mask]; v = hsv[:, :, 2][mask]
    hmean = float(hue(h))
    smean = float(np.mean(s))
    vmean = float(np.mean(v))
    return hmean, smean, vmean

"""Calculating the color bound of bad apples and good apples
   Successfully calculated, the result of calculating is saved as gupper, glower, bupper, blower"""

# # good apple hsv bound

# h_means, s_means, v_means = [], [], []
# i = 1
# for path in goodli:
#     img = process(path)

#     alpha = img[:, :, 3]
#     mask = alpha > 0
#     bgr = img[:, :, :3]
#     hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

#     h = hsv[:, :, 0][mask]
#     s = hsv[:, :, 1][mask]
#     v = hsv[:, :, 2][mask]

#     if len(h) > 0:
#         h_r = hue(h)
#         s_r = np.mean(s)
#         v_r = np.mean(v)
#         h_means.append(h_r)
#         s_means.append(s_r)
#         v_means.append(v_r)

#     print(f"Successfully calculated: Number {i}\nType: Good")
#     print(f"H: {h_r} S: {s_r} V: {v_r}")
#     i += 1

# avg_h = hue(h_means)
# left, right = divide(h_means)
# if len(left) == 0:
#     min_h = np.min(right)
#     max_h = np.max(right)
# elif len(right) == 0:
#     min_h = np.min(left)
#     max_h = np.max(left)
# else:
#     min_h = np.min(right)
#     max_h = np.max(left)
# avg_s = np.mean(s_means); max_s = np.max(s_means); min_s = np.min(s_means)
# avg_v = np.mean(v_means); max_v = np.max(v_means); min_v = np.min(v_means)

# glower = [mix(avg_h, min_h), (avg_s + 2 * min_s) / 3, (avg_v + 2 * min_v) / 3]
# gupper = [mix(avg_h, max_h), (avg_s + 2 * max_s) / 3, (avg_v + 2 * max_v) / 3]
# # bad apple hsv bound

# hmeans, smeans, vmeans = [], [], []
# i = 1
# for path in badli:
#     img = process(path)
#     alpha = img[:, :, 3]
#     mask = alpha > 0
#     bgr = img[:, :, :3]
#     hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

#     h = hsv[:, :, 0][mask]
#     s = hsv[:, :, 1][mask]
#     v = hsv[:, :, 2][mask]

#     if len(h) > 0:
#         hr = hue(h)
#         sr = np.mean(s)
#         vr = np.mean(v)
#         hmeans.append(hr)
#         smeans.append(sr)
#         vmeans.append(vr)
#     print(f"Successfully calculated: Number {i}\nType: Bad")
#     print(f"H: {hr} S: {sr} V: {vr}")
#     i += 1

# left, right = divide(hmeans)
# avgh = hue(hmeans)
# if len(left) == 0:
#     minh = np.min(right)
#     maxh = np.max(right)
# elif len(right) == 0:
#     minh = np.min(left)
#     maxh = np.max(left)
# else:
#     minh = np.min(right)
#     maxh = np.max(left)
# avgs = np.mean(smeans); maxs = np.max(smeans); mins = np.min(smeans)
# avgv = np.mean(vmeans); maxv = np.max(vmeans); minv = np.min(vmeans)
# blower = [mix(avgh, minh), (avgs + 2 * mins) / 3, (avgv + 2 * minv) / 3]
# bupper = [mix(avgh, maxh), (avgs + 2 * maxs) / 3, (avgv + 2 * maxv) / 3]


# print(f"Good upper bound: {tuple(map(int, gupper))}")
# print(f"Good lower bound: {tuple(map(int, glower))}")
# print(f"Bad upper bound: {tuple(map(int, bupper))}")
# print(f"Bad lower bound: {tuple(map(int, blower))}")

gupper = (29, 217, 198)
glower = (14, 84, 77)
bupper = (37, 179, 185)
blower = (17, 84, 91)

# Don't have to divide the color bound 
# 14 ~ 29, 17 ~ 37

def h_circ_dist(h1, h2):
    """두 H값을 단위벡터 사잇각으로 거리 계산 (hue()와 동일한 방식)"""
    rad1 = np.deg2rad(h1 * 2)
    rad2 = np.deg2rad(h2 * 2)
    dot = np.cos(rad1) * np.cos(rad2) + np.sin(rad1) * np.sin(rad2)
    dot = np.clip(dot, -1.0, 1.0)
    return np.rad2deg(np.arccos(dot)) / 2


def _norm_dist(hsv, lower, upper):
    h, s, v = hsv[0], hsv[1], hsv[2]

    h_center = hue([lower[0], upper[0]])
    h_half   = h_circ_dist(lower[0], upper[0]) / 2

    s_center = (lower[1] + upper[1]) / 2
    s_half   = (upper[1] - lower[1]) / 2

    v_center = (lower[2] + upper[2]) / 2
    v_half   = (upper[2] - lower[2]) / 2

    dh = h_circ_dist(h, h_center) / h_half if h_half > 0 else float('inf')
    ds = abs(s - s_center) / s_half         if s_half  > 0 else float('inf')
    dv = abs(v - v_center) / v_half         if v_half  > 0 else float('inf')

    return max(dh, ds, dv)


def sortApple(hsv: list, gupper=gupper, glower=glower, bupper=bupper, blower=blower):
    d_good = _norm_dist(hsv, glower, gupper)
    d_bad  = _norm_dist(hsv, blower, bupper)

    in_good = d_good <= 1
    in_bad  = d_bad  <= 1

    if in_good and in_bad:
        return "Good" if d_good <= d_bad else "Bad"
    elif in_good:
        return "Good"
    elif in_bad:
        return "Bad"
    else:
        return "Imperfect"

if __name__ == "__main__":
# Evaluating the algorithm
    import random

    tot = 0
    cnt = 0

    li = []
    for j in range(len(goodli)):
        li.append({"Path": goodli[j], "Type": "Good"})
    for j in range(len(badli)):
        li.append({"Path": badli[j], "Type": "Bad"})
    random.shuffle(li)
    for i in range(200):
        path = li[i]["Path"]
        h, s, v = hsvcal(path)
        type = li[i]["Type"]
        hsv = [h, s, v]
        typeeval = sortApple(hsv)
        if typeeval == type:
            cnt += 1
        tot += 1
        if tot % 10 == 0:
            print(f"Number {tot}: Successfully calculated")

    print(f"Accuracy: {100 * cnt / tot}")
