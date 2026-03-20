from __future__ import annotations

import math

import numpy as np

OUTER_LIP_LANDMARKS = [
    61,
    146,
    91,
    181,
    84,
    17,
    314,
    405,
    321,
    375,
    291,
    409,
    270,
    269,
    267,
    0,
]
INNER_LIP_LANDMARKS = [
    78,
    95,
    88,
    178,
    87,
    14,
    317,
    402,
    318,
    324,
    308,
    415,
    310,
    311,
    312,
    13,
]
FACE_OVAL = [
    10,
    338,
    297,
    332,
    284,
    251,
    389,
    356,
    454,
    323,
    361,
    288,
    397,
    365,
    379,
    378,
    400,
    377,
    152,
    148,
    176,
    149,
    150,
    136,
    172,
    58,
    132,
    93,
    234,
    127,
    162,
    21,
    54,
    103,
    67,
    109,
]
LEFT_FACE = 234
RIGHT_FACE = 454
TOP_FACE = 10
BOTTOM_FACE = 152


def polygon_area(vertices: np.ndarray) -> float:
    x = vertices[:, 0]
    y = vertices[:, 1]
    return float(
        0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
    )


def polygon_perimeter(vertices: np.ndarray) -> float:
    perimeter = 0.0
    for idx in range(len(vertices)):
        perimeter += float(
            np.linalg.norm(vertices[idx] - vertices[(idx + 1) % len(vertices)])
        )
    return perimeter


def extract_lip_metrics(landmarks: np.ndarray) -> dict[str, float]:
    outer_lip = np.array([landmarks[i] for i in OUTER_LIP_LANDMARKS])
    inner_lip = np.array([landmarks[i] for i in INNER_LIP_LANDMARKS])
    face_oval = np.array([landmarks[i] for i in FACE_OVAL])

    face_width = abs(float(landmarks[RIGHT_FACE][0] - landmarks[LEFT_FACE][0]))
    face_height = abs(float(landmarks[BOTTOM_FACE][1] - landmarks[TOP_FACE][1]))

    face_area = polygon_area(face_oval)
    outer_lip_area = polygon_area(outer_lip)
    inner_lip_area = polygon_area(inner_lip)
    lip_area = outer_lip_area - inner_lip_area
    area_ratio = lip_area / face_area if face_area > 0 else math.nan

    outer_lip_y = outer_lip[:, 1]
    outer_lip_x = outer_lip[:, 0]
    lip_height = float(max(outer_lip_y) - min(outer_lip_y))
    outer_lip_width = float(max(outer_lip_x) - min(outer_lip_x))

    inner_lip_x = inner_lip[:, 0]
    inner_lip_width = float(max(inner_lip_x) - min(inner_lip_x))
    total_width = outer_lip_width + inner_lip_width

    top_lip_bottom = float(landmarks[13][1])
    bottom_lip_top = float(landmarks[14][1])
    lip_openness = bottom_lip_top - top_lip_bottom

    outer_perimeter = polygon_perimeter(outer_lip)
    circularity = (
        float(4.0 * math.pi * lip_area / (outer_perimeter * outer_perimeter))
        if outer_perimeter > 0
        else math.nan
    )

    normalized_height = lip_height / face_height if face_height > 0 else math.nan
    normalized_outer_width = (
        outer_lip_width / face_width if face_width > 0 else math.nan
    )
    normalized_inner_width = (
        inner_lip_width / face_width if face_width > 0 else math.nan
    )
    normalized_total_width = total_width / face_width if face_width > 0 else math.nan
    normalized_open = lip_openness / face_height if face_height > 0 else math.nan

    return {
        "area": area_ratio,
        "face_width": face_width,
        "face_height": face_height,
        "height_px": lip_height,
        "outer_width_px": outer_lip_width,
        "inner_width_px": inner_lip_width,
        "total_width_px": total_width,
        "open_px": lip_openness,
        "length": outer_perimeter,
        "height": normalized_height,
        "outer_width": normalized_outer_width,
        "inner_width": normalized_inner_width,
        "total_width": normalized_total_width,
        "open": normalized_open,
        "circularity": circularity,
    }


def lip_extract(landmarks: np.ndarray) -> dict[str, float]:
    return extract_lip_metrics(landmarks)
