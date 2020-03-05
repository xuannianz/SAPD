import tensorflow as tf


def xyxy2cxcywh(xyxy):
    """
    Convert [x1 y1 x2 y2] box format to [xc yc w h] format.
    """
    return tf.concat((0.5 * (xyxy[:, 0:2] + xyxy[:, 2:4]), xyxy[:, 2:4] - xyxy[:, 0:2]), axis=-1)


def cxcywh2xyxy(xywh):
    """
    Convert [cx cy w y] box format to [x1 y1 x2 y2] format.
    """
    return tf.concat((xywh[:, 0:2] - 0.5 * xywh[:, 2:4], xywh[:, 0:2] + 0.5 * xywh[:, 2:4]), axis=-1)


def normalize_boxes(boxes, width, height, stride):
    # normalize:
    x1 = boxes[:, 0:1] / stride / width
    y1 = boxes[:, 1:2] / stride / height
    x2 = boxes[:, 2:3] / stride / width
    y2 = boxes[:, 3:4] / stride / height
    return tf.concat([y1, x1, y2, x2], axis=-1)


def shrink_and_normalize_boxes(boxes, width, height, stride, shrink_ratio=0.2):
    # shrink
    boxes = xyxy2cxcywh(boxes)
    boxes = tf.concat((boxes[:, :2], boxes[:, 2:] * shrink_ratio), axis=-1)
    boxes = cxcywh2xyxy(boxes)
    # normalize:
    x1 = boxes[:, 0:1] / stride / width
    y1 = boxes[:, 1:2] / stride / height
    x2 = boxes[:, 2:3] / stride / width
    y2 = boxes[:, 3:4] / stride / height
    return tf.concat([x1, y1, x2, y2], axis=-1)


def shrink_and_project_boxes(boxes, width, height, stride, shrink_ratio=0.2, keep_dims=False):
    """
    Compute proportional box coordinates.

    Box centers are fixed. Box w and h scaled by scale.
    """
    # shrink
    boxes = xyxy2cxcywh(boxes)
    boxes = tf.concat((boxes[:, :2], boxes[:, 2:] * shrink_ratio), axis=-1)
    boxes = cxcywh2xyxy(boxes)

    if keep_dims:
        x1 = tf.floor(boxes[:, 0:1] / stride)
        y1 = tf.floor(boxes[:, 1:2] / stride)
        x2 = tf.math.ceil(boxes[:, 2:3] / stride)
        y2 = tf.math.ceil(boxes[:, 3:4] / stride)
    else:
        x1 = tf.floor(boxes[:, 0] / stride)
        y1 = tf.floor(boxes[:, 1] / stride)
        x2 = tf.math.ceil(boxes[:, 2] / stride)
        y2 = tf.math.ceil(boxes[:, 3] / stride)
    width = tf.cast(width, tf.float32)
    height = tf.cast(height, tf.float32)
    x2 = tf.cast(tf.clip_by_value(x2, 1, width), tf.int32)
    y2 = tf.cast(tf.clip_by_value(y2, 1, height), tf.int32)
    x1 = tf.cast(tf.clip_by_value(x1, 0, tf.cast(x2, tf.float32) - 1), tf.int32)
    y1 = tf.cast(tf.clip_by_value(y1, 0, tf.cast(y2, tf.float32) - 1), tf.int32)

    return x1, y1, x2, y2


def trim_padding_boxes(boxes):
    """
    Often boxes are represented with matrices of shape [N, 4] and are padded with zeros.
    This removes zero boxes.

    Args:
        boxes: [N, 4] matrix of boxes.

    Returns:

    """
    non_zeros = tf.cast(tf.reduce_sum(tf.abs(boxes), axis=1), tf.bool)
    boxes = tf.boolean_mask(boxes, non_zeros)
    return boxes, non_zeros
