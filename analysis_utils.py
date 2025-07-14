import cv2
import numpy as np

def score_redness(region):
    if region is None or region.size == 0:
        return 0

    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    hue, sat, _ = cv2.split(hsv)

    red_mask = (hue < 10) | (hue > 160)
    if np.sum(red_mask) == 0:
        return 0

    red_score = np.mean(sat[red_mask])
    return min(int((red_score / 255) * 10), 10)

def score_shine(region):
    if region is None or region.size == 0:
        return 0

    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    bright_pixels = np.mean(gray)
    return min(int((bright_pixels / 255) * 10), 10)

def score_texture(region):
    if region is None or region.size == 0:
        return 0

    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    score = np.sum(edges) / 255 / (region.shape[0] * region.shape[1])
    return min(int(score * 50), 10)

def score_pores(region):
    if region is None or region.size == 0:
        return 0

    blur = cv2.GaussianBlur(region, (5, 5), 0)
    diff = cv2.absdiff(region, blur)
    score = np.mean(diff)
    return min(int((score / 50) * 10), 10)

def score_dark_circle(region):
    if region is None or region.size == 0:
        return 0

    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    shadow = 255 - np.mean(gray)
    return min(int((shadow / 255) * 10), 10)
