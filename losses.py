"""
Copyright 2017-2018 Fizyr (https://fizyr.com)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

# import keras
from tensorflow import keras
import tensorflow as tf


def focal(alpha=0.25, gamma=2.0):
    """
    Create a functor for computing the focal loss.

    Args
        alpha: Scale the focal weight with alpha.
        gamma: Take the power of the focal weight with gamma.

    Returns
        A functor that computes the focal loss using the alpha and gamma.
    """

    def _focal(y_true, y_pred):
        """
        Compute the focal loss given the target tensor and the predicted tensor.

        As defined in https://arxiv.org/abs/1708.02002

        Args
            y_true: Tensor of target data from the generator with shape (B, N, num_classes).
            y_pred: Tensor of predicted data from the network with shape (B, N, num_classes).

        Returns
            The focal loss of y_pred w.r.t. y_true.
        """
        # compute the focal loss
        alpha_factor = tf.ones_like(y_true) * alpha
        alpha_factor = tf.where(tf.equal(y_true, 1), alpha_factor, 1 - alpha_factor)
        focal_weight = tf.where(keras.backend.equal(y_true, 1), 1 - y_pred, y_pred)
        focal_weight = alpha_factor * focal_weight ** gamma
        cls_loss = focal_weight * tf.keras.backend.binary_crossentropy(y_true, y_pred)

        # compute the normalizer: the number of positive anchors
        normalizer = tf.cast(tf.shape(y_pred)[1], dtype=tf.float32)
        normalizer = tf.maximum(1.0, normalizer)

        return tf.reduce_sum(cls_loss) / normalizer

    return _focal


def iou():
    def _iou(y_true, y_pred):
        y_true = tf.maximum(y_true, 0)
        pred_left = y_pred[:, :, 0]
        pred_top = y_pred[:, :, 1]
        pred_right = y_pred[:, :, 2]
        pred_bottom = y_pred[:, :, 3]

        # (num_pos, )
        target_left = y_true[:, :, 0]
        target_top = y_true[:, :, 1]
        target_right = y_true[:, :, 2]
        target_bottom = y_true[:, :, 3]

        target_area = (target_left + target_right) * (target_top + target_bottom)
        pred_area = (pred_left + pred_right) * (pred_top + pred_bottom)
        w_intersect = tf.minimum(pred_left, target_left) + tf.minimum(pred_right, target_right)
        h_intersect = tf.minimum(pred_bottom, target_bottom) + tf.minimum(pred_top, target_top)

        area_intersect = w_intersect * h_intersect
        area_union = target_area + pred_area - area_intersect

        # (num_pos, )
        iou_loss = -tf.math.log((area_intersect + 1e-7) / (area_union + 1e-7))
        # compute the normalizer: the number of positive anchors
        normalizer = tf.maximum(1, tf.reduce_prod(tf.shape(y_true)[0:2]))
        normalizer = tf.cast(normalizer, dtype=tf.float32)
        return tf.reduce_sum(iou_loss) / normalizer

    return _iou


def smooth_l1(sigma=3.0):
    """
    Create a smooth L1 loss functor.
    Args
        sigma: This argument defines the point where the loss changes from L2 to L1.
    Returns
        A functor for computing the smooth L1 loss given target data and predicted data.
    """
    sigma_squared = sigma ** 2

    def _smooth_l1(y_true, y_pred):
        """ Compute the smooth L1 loss of y_pred w.r.t. y_true.
        Args
            y_true: Tensor from the generator of shape (B, N, 5). The last value for each box is the state of the anchor (ignore, negative, positive).
            y_pred: Tensor from the network of shape (B, N, 4).
        Returns
            The smooth L1 loss of y_pred w.r.t. y_true.
        """
        # separate target and state
        regression = y_pred
        regression_target = y_true[:, :, :-1]
        anchor_state = y_true[:, :, -1]

        # filter out "ignore" anchors
        indices = tf.where(keras.backend.equal(anchor_state, 1))
        regression = tf.gather_nd(regression, indices)
        regression_target = tf.gather_nd(regression_target, indices)

        # compute smooth L1 loss
        # f(x) = 0.5 * (sigma * x)^2          if |x| < 1 / sigma / sigma
        #        |x| - 0.5 / sigma / sigma    otherwise
        regression_diff = regression - regression_target
        regression_diff = keras.backend.abs(regression_diff)
        regression_loss = tf.where(
            keras.backend.less(regression_diff, 1.0 / sigma_squared),
            0.5 * sigma_squared * keras.backend.pow(regression_diff, 2),
            regression_diff - 0.5 / sigma_squared
        )

        # compute the normalizer: the number of positive anchors
        normalizer = keras.backend.maximum(1, keras.backend.shape(indices)[0])
        normalizer = keras.backend.cast(normalizer, dtype=keras.backend.floatx())
        return keras.backend.sum(regression_loss) / normalizer

    return _smooth_l1


def focal_with_weight_and_mask(alpha=0.25, gamma=2.0):
    """
    Create a functor for computing the focal loss.

    Args
        alpha: Scale the focal weight with alpha.
        gamma: Take the power of the focal weight with gamma.

    Returns
        A functor that computes the focal loss using the alpha and gamma.
    """

    def _focal(inputs):
        """
        Compute the focal loss given the target tensor and the predicted tensor.

        As defined in https://arxiv.org/abs/1708.02002

        Args
            y_true: Tensor of target data from the generator with shape (B, N, num_classes).
            y_pred: Tensor of predicted data from the network with shape (B, N, num_classes + 2).

        Returns
            The focal loss of y_pred w.r.t. y_true.
        """
        # compute the focal loss
        y_true, y_pred, soft_weight, mask = inputs[0][..., :-2], inputs[1], inputs[0][..., -2], inputs[0][..., -1]
        alpha_factor = keras.backend.ones_like(y_true) * alpha
        alpha_factor = tf.where(keras.backend.equal(y_true, 1), alpha_factor, 1 - alpha_factor)
        focal_weight = tf.where(keras.backend.equal(y_true, 1), 1 - y_pred, y_pred)
        focal_weight = alpha_factor * focal_weight ** gamma
        soft_weight = tf.expand_dims(soft_weight, axis=-1)
        cls_loss = focal_weight * soft_weight * tf.keras.backend.binary_crossentropy(y_true, y_pred)
        # compute the normalizer: the number of positive locations
        num_pos = tf.reduce_sum(mask * soft_weight[..., 0])
        normalizer = tf.maximum(1.0, tf.cast(num_pos, dtype=tf.float32))
        return tf.reduce_sum(cls_loss) / normalizer

    return _focal


def iou_with_weight_and_mask():
    def _iou(inputs):
        y_true, y_pred, soft_weight, mask = inputs[0][..., :4], inputs[1], inputs[0][..., 4], inputs[0][..., 5]
        y_true = tf.maximum(y_true, 0)
        pred_left = y_pred[:, :, 0]
        pred_top = y_pred[:, :, 1]
        pred_right = y_pred[:, :, 2]
        pred_bottom = y_pred[:, :, 3]

        target_left = y_true[:, :, 0]
        target_top = y_true[:, :, 1]
        target_right = y_true[:, :, 2]
        target_bottom = y_true[:, :, 3]

        target_area = (target_left + target_right) * (target_top + target_bottom)
        masked_target_area = tf.boolean_mask(target_area, mask)
        pred_area = (pred_left + pred_right) * (pred_top + pred_bottom)
        masked_pred_area = tf.boolean_mask(pred_area, mask)
        w_intersect = tf.minimum(pred_left, target_left) + tf.minimum(pred_right, target_right)
        h_intersect = tf.minimum(pred_bottom, target_bottom) + tf.minimum(pred_top, target_top)

        area_intersect = w_intersect * h_intersect
        masked_area_intersect = tf.boolean_mask(area_intersect, mask)
        masked_area_union = masked_target_area + masked_pred_area - masked_area_intersect

        masked_soft_weight = tf.boolean_mask(soft_weight, mask)
        # (B, N)
        masked_iou_loss = -tf.math.log((masked_area_intersect + 1e-7) / (masked_area_union + 1e-7)) * masked_soft_weight

        # compute the normalizer: the number of positive locations
        num_pos = tf.reduce_sum(mask * soft_weight)
        normalizer = keras.backend.maximum(1., num_pos)
        return tf.reduce_sum(masked_iou_loss) / normalizer

    return _iou
