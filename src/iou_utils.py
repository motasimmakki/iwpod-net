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

import numpy as np

def extract_quad_from_output(pred):
    # pred: shape (13, 13, 9), per grid cell has [obj, tx, ty, tw, th, a00, a01, a10, a11]
    # Find the most confident cell
    max_pos = np.unravel_index(np.argmax(pred[..., 0]), pred[..., 0].shape)

    # Extract the parameters
    pred_cell = pred[max_pos]

    # Fix here — ensure safe max on arrays
    affinex = [np.maximum(pred_cell[1], 0.), pred_cell[2], pred_cell[3]]
    affiney = [pred_cell[4], np.maximum(pred_cell[5], 0.), pred_cell[6]]

    # Define quadrilateral corners from affine transform (similar to your loc_loss logic)
    v = 0.5
    base = np.array([[-v, -v, 1],
                     [ v, -v, 1],
                     [ v,  v, 1],
                     [-v,  v, 1]])  # shape (4,3)

    quad = []
    for i in range(4):
        x = np.dot(affinex, base[i])
        y = np.dot(affiney, base[i])
        quad.append([x * 208, y * 208])  # scale to image size if needed

    return np.array(quad)  # shape (4,2)

