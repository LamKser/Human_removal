import base64
import zlib

import numpy as np
import cv2

from imantics import Mask


def poly_area(points):
    return 0.5 * np.abs(np.dot(points[:, 0], np.roll(points[:, 1], 1))\
            - np.dot(points[:, 1], np.roll(points[:, 0], 1)))


def base64_2_mask(s):
    z = zlib.decompress(base64.b64decode(s))
    n = np.frombuffer(z, np.uint8)
    mask = cv2.imdecode(n, cv2.IMREAD_UNCHANGED)[:, :, 3].astype(bool)
    return mask


def xywh_2_xyxy(points, height, width):
    xmin, ymin = points
    xmax, ymax = xmin + width, ymin + height
    return xmin, ymin, xmax, ymax


def mask_2_polygons(mask):
    polygons = Mask(mask).polygons()
    max_dict = dict()
    for poly in polygons.points:
        area = poly_area(poly)
        max_dict[area] = poly
    idx = max(max_dict)
    return max_dict[idx]


def point_2_str(points, height, width):
    str_point = ''
    for point in points:
        str_point = str_point + f'{point[0] / width} {point[1] / height} '
    return str_point.rstrip()