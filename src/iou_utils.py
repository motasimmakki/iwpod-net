import numpy as np
from shapely.geometry import Polygon

def compute_qiou(pts_true, pts_pred):
    """Compute IoU between two quadrilaterals using shapely."""
    poly1 = Polygon(pts_true)
    poly2 = Polygon(pts_pred)

    if not poly1.is_valid or not poly2.is_valid:
        return 0.0

    inter_area = poly1.intersection(poly2).area
    union_area = poly1.union(poly2).area

    return inter_area / union_area if union_area > 0 else 0.0

def extract_quad_from_output(pred):
    affinex = [max(pred[1], 0.), pred[2], pred[3]]
    affiney = [pred[4], max(pred[5], 0.), pred[6]]
    affine_matrix_x = np.array(affinex)
    affine_matrix_y = np.array(affiney)

    v = 0.5
    base = np.array([
        [-v, -v, 1],
        [ v, -v, 1],
        [ v,  v, 1],
        [-v,  v, 1]
    ])

    x_coords = np.dot(base, affine_matrix_x)
    y_coords = np.dot(base, affine_matrix_y)
    return np.stack([x_coords, y_coords], axis=1)  # shape: (4,2)
