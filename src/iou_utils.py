def compute_iou(box1, box2):
    """
    Compute IoU between two bounding boxes.
    box format: [x1, y1, x2, y2]
    """
    xi1 = max(box1[0], box2[0])
    yi1 = max(box1[1], box2[1])
    xi2 = min(box1[2], box2[2])
    yi2 = min(box1[3], box2[3])
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area

    iou = inter_area / union_area if union_area != 0 else 0
    return iou

def extract_box_from_output(pred):
    # pred: shape (13, 13, 9)
    # Example: Find the most confident grid cell and use center/size to construct bounding box

    max_pos = np.unravel_index(np.argmax(pred[...,0]), pred[...,0].shape)
    cx = (max_pos[1] + pred[max_pos][1]) * 16  # x offset * stride
    cy = (max_pos[0] + pred[max_pos][2]) * 16  # y offset * stride
    w  = pred[max_pos][3] * 208
    h  = pred[max_pos][4] * 208

    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return [x1, y1, x2, y2]