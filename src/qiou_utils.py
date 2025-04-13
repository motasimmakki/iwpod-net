import numpy as np
from shapely.geometry import Polygon

# def compute_qiou(pred_pts, gt_pts):
#     pred_poly = Polygon(pred_pts)
#     gt_poly = Polygon(gt_pts)

#     if not pred_poly.is_valid or not gt_poly.is_valid:
#         return 0.0

#     inter_area = pred_poly.intersection(gt_poly).area
#     union_area = pred_poly.union(gt_poly).area

#     return inter_area / union_area if union_area != 0 else 0.0

def compute_qiou(pred_pts, gt_pts):
    pred_pts = np.array(pred_pts).reshape(-1, 2)
    gt_pts = np.array(gt_pts).reshape(-1, 2)

    pred_poly = Polygon(pred_pts)
    gt_poly = Polygon(gt_pts)

    if not pred_poly.is_valid or not gt_poly.is_valid:
        return 0.0

    inter_area = pred_poly.intersection(gt_poly).area
    union_area = pred_poly.union(gt_poly).area

    return inter_area / union_area if union_area != 0 else 0.0


def load_ground_truth(txt_path, img_shape):
    with open(txt_path, 'r') as f:
        line = f.readline().strip()
        parts = line.split(',')

    # Extract 8 coordinates (x1, x2, ..., x4), (y1, ..., y4)
    coords = list(map(float, parts[1:9]))
    xs = coords[:4]
    ys = coords[4:8]

    w, h = img_shape[1], img_shape[0]
    points = np.array([[xs[i] * w, ys[i] * h] for i in range(4)], dtype=np.float32)
    return points
