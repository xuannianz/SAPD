from tensorflow.keras.layers import Layer
# from keras.layers import Layer
import tensorflow as tf
from util_graphs import trim_padding_boxes, normalize_boxes, shrink_and_project_boxes
from losses import focal, iou


class MetaSelectInput(Layer):
    def __init__(self, strides=(8, 16, 32, 64, 128), pool_size=7, **kwargs):
        self.strides = strides
        self.pool_size = pool_size
        super(MetaSelectInput, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        batch_gt_boxes = inputs[0][..., :4]
        list_batch_fms = inputs[1: 1 + len(self.strides)]
        batch_size = tf.shape(batch_gt_boxes)[0]
        max_gt_boxes = tf.shape(batch_gt_boxes)[1]
        gt_boxes_batch_ids = tf.tile(tf.expand_dims(tf.range(batch_size), axis=-1), (1, max_gt_boxes))
        gt_boxes_batch_ids = tf.reshape(gt_boxes_batch_ids, (-1,))
        batch_gt_boxes = tf.reshape(batch_gt_boxes, (-1, tf.shape(batch_gt_boxes)[-1]))

        # (total_num_gt_boxes, )
        gt_boxes, non_zeros = trim_padding_boxes(batch_gt_boxes)
        gt_boxes_batch_ids = tf.boolean_mask(gt_boxes_batch_ids, non_zeros)

        rois_from_fms = []
        for i, batch_fm in enumerate(list_batch_fms):
            stride = tf.constant(self.strides[i], dtype=tf.float32)
            fm_height = tf.cast(tf.shape(batch_fm)[1], dtype=tf.float32)
            fm_width = tf.cast(tf.shape(batch_fm)[2], dtype=tf.float32)
            normed_gt_boxes = normalize_boxes(gt_boxes, width=fm_width, height=fm_height, stride=stride)
            rois = tf.image.crop_and_resize(batch_fm, normed_gt_boxes, gt_boxes_batch_ids,
                                            (self.pool_size, self.pool_size))
            rois_from_fms.append(rois)
        rois = tf.concat(rois_from_fms, axis=-1)
        return rois, gt_boxes_batch_ids

    def compute_output_shape(self, input_shape):
        return [[None, self.pool_size, self.pool_size, None], [None, ]]

    def get_config(self):
        """
        Gets the configuration of this layer.

        Returns
            Dictionary containing the parameters of this layer.
        """
        config = super(MetaSelectInput, self).get_config()
        config.update(strides=self.strides, pool_size=self.pool_size)
        return config


def build_meta_select_target(cls_pred, regr_pred, gt_boxes, feature_shapes, strides, shrink_ratio=0.2):
    gt_labels = tf.cast(gt_boxes[:, 4], tf.int32)
    gt_boxes = gt_boxes[:, :4]
    max_gt_boxes = tf.shape(gt_boxes)[0]
    focal_loss = focal()
    iou_loss = iou()
    gt_boxes, non_zeros = trim_padding_boxes(gt_boxes)
    num_gt_boxes = tf.shape(gt_boxes)[0]
    gt_labels = tf.boolean_mask(gt_labels, non_zeros)
    level_losses = []
    for level_id in range(len(strides)):
        stride = strides[level_id]
        fh = feature_shapes[level_id][0]
        fw = feature_shapes[level_id][1]
        fa = tf.reduce_prod(feature_shapes, axis=-1)
        start_idx = tf.reduce_sum(fa[:level_id])
        end_idx = start_idx + fh * fw
        cls_pred_i = tf.reshape(cls_pred[start_idx:end_idx], (fh, fw, tf.shape(cls_pred)[-1]))
        regr_pred_i = tf.reshape(regr_pred[start_idx:end_idx], (fh, fw, tf.shape(regr_pred)[-1]))
        # (num_gt_boxes, )
        x1, y1, x2, y2 = shrink_and_project_boxes(gt_boxes, fw, fh, stride, shrink_ratio=shrink_ratio)

        def compute_gt_box_loss(args):
            x1_ = args[0]
            y1_ = args[1]
            x2_ = args[2]
            y2_ = args[3]
            gt_box = args[4]
            gt_label = args[5]

            def do_match_pixels_in_level():
                locs_cls_pred_i = cls_pred_i[y1_:y2_, x1_:x2_, :]
                locs_cls_pred_i = tf.reshape(locs_cls_pred_i, (-1, tf.shape(locs_cls_pred_i)[-1]))
                locs_cls_true_i = tf.zeros_like(locs_cls_pred_i)
                gt_label_col = tf.ones_like(locs_cls_true_i[:, 0:1])
                locs_cls_true_i = tf.concat([locs_cls_true_i[:, :gt_label],
                                             gt_label_col,
                                             locs_cls_true_i[:, gt_label + 1:],
                                             ], axis=-1)
                loss_cls = focal_loss(tf.expand_dims(locs_cls_true_i, axis=0), tf.expand_dims(locs_cls_pred_i, axis=0))
                locs_regr_pred_i = regr_pred_i[y1_:y2_, x1_:x2_, :]
                locs_regr_pred_i = tf.reshape(locs_regr_pred_i, (-1, tf.shape(locs_regr_pred_i)[-1]))
                locs_x = tf.cast(tf.range(x1_, x2_), dtype=tf.float32)
                locs_y = tf.cast(tf.range(y1_, y2_), dtype=tf.float32)
                shift_x = (locs_x + 0.5) * stride
                shift_y = (locs_y + 0.5) * stride
                shift_xx, shift_yy = tf.meshgrid(shift_x, shift_y)
                shift_xx = tf.reshape(shift_xx, (-1,))
                shift_yy = tf.reshape(shift_yy, (-1,))
                shifts = tf.stack((shift_xx, shift_yy, shift_xx, shift_yy), axis=-1)
                l = tf.maximum(shifts[:, 0] - gt_box[0], 0)
                t = tf.maximum(shifts[:, 1] - gt_box[1], 0)
                r = tf.maximum(gt_box[2] - shifts[:, 2], 0)
                b = tf.maximum(gt_box[3] - shifts[:, 3], 0)
                locs_regr_true_i = tf.stack([l, t, r, b], axis=-1)
                locs_regr_true_i = locs_regr_true_i / 4.0 / stride
                loss_regr = iou_loss(tf.expand_dims(locs_regr_true_i, axis=0), tf.expand_dims(locs_regr_pred_i, axis=0))
                return loss_cls + loss_regr

            def do_not_match_pixels_in_level():
                box_loss = tf.constant(1e7, dtype=tf.float32)
                return box_loss

            level_box_loss = tf.cond(
                tf.equal(tf.cast(x1_, tf.int32), tf.cast(x2_, tf.int32)) |
                tf.equal(tf.cast(y1_, tf.int32), tf.cast(y2_, tf.int32)),
                do_not_match_pixels_in_level,
                do_match_pixels_in_level
            )
            return level_box_loss

        level_loss = tf.map_fn(
            compute_gt_box_loss,
            elems=[x1, y1, x2, y2, gt_boxes, gt_labels],
            dtype=tf.float32
        )
        level_losses.append(level_loss)
    losses = tf.stack(level_losses, axis=-1)
    gt_box_levels = tf.argmin(losses, axis=-1, output_type=tf.int32)
    padding_gt_box_levels = tf.ones((max_gt_boxes - num_gt_boxes), dtype=tf.int32) * -1
    gt_box_levels = tf.concat([gt_box_levels, padding_gt_box_levels], axis=0)
    return gt_box_levels


class MetaSelectTarget(Layer):
    def __init__(self, strides=(8, 16, 32, 64, 128), shrink_ratio=0.2, **kwargs):
        self.strides = strides
        self.shrink_ratio = shrink_ratio
        super(MetaSelectTarget, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        batch_cls_pred = inputs[0]
        batch_regr_pred = inputs[1]
        feature_shapes = inputs[2][0]
        batch_gt_boxes = inputs[3]

        def _build_meta_select_target(args):
            cls_pred = args[0]
            regr_pred = args[1]
            gt_boxes = args[2]

            return build_meta_select_target(
                cls_pred,
                regr_pred,
                gt_boxes,
                feature_shapes=feature_shapes,
                strides=self.strides,
                shrink_ratio=self.shrink_ratio,
            )

        # (b, MAX_GT_BOXES)
        batch_box_levels = tf.map_fn(
            _build_meta_select_target,
            elems=[batch_cls_pred, batch_regr_pred, batch_gt_boxes],
            dtype=tf.int32,
        )
        batch_box_levels = tf.reshape(batch_box_levels, (-1,))
        mask = tf.not_equal(batch_box_levels, -1)
        valid_box_levels = tf.boolean_mask(batch_box_levels, mask)
        return valid_box_levels

    def compute_output_shape(self, input_shape):
        return None,

    def get_config(self):
        """
        Gets the configuration of this layer.

        Returns
            Dictionary containing the parameters of this layer.
        """
        config = super(MetaSelectTarget, self).get_config()
        config.update(strides=self.strides, shrink_ratio=self.shrink_ratio)
        return config


class MetaSelectWeight(Layer):
    def __init__(self, max_gt_boxes=100, soft_select=True, batch_size=32, **kwargs):
        self.max_gt_boxes = max_gt_boxes
        self.soft_select = soft_select
        self.batch_size = batch_size
        super(MetaSelectWeight, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        if self.soft_select:
            gt_boxes_select_weight = inputs[0]
        else:
            gt_boxes_select_weight = tf.one_hot(inputs[0], 5)
        gt_boxes_batch_ids = inputs[1]
        # (b, 1) --> (b, )
        batch_num_gt_boxes = inputs[2][:, 0]

        batch_select_weight = []
        for i in range(self.batch_size):
            batch_item_select_weight = tf.boolean_mask(gt_boxes_select_weight, tf.equal(gt_boxes_batch_ids, i))
            pad_top_bot = tf.stack([tf.constant(0), self.max_gt_boxes - batch_num_gt_boxes[i]], axis=0)
            pad = tf.stack([pad_top_bot, tf.constant([0, 0])], axis=0)
            batch_select_weight.append(tf.pad(batch_item_select_weight, pad, constant_values=-1))
        batch_select_weight = tf.stack(batch_select_weight, axis=0)
        return batch_select_weight

    def compute_output_shape(self, input_shapes):
        return input_shapes[1][0], self.max_gt_boxes, 5

    def get_config(self):
        base_config = super(MetaSelectWeight, self).get_config()
        base_config.update(max_gt_boxes=self.max_gt_boxes, soft_select=self.soft_select)
        return base_config


def build_sapd_target(gt_boxes, meta_select_weight, fm_shapes, num_classes, strides, shrink_ratio=0.2):
    gt_labels = tf.cast(gt_boxes[:, 4], tf.int32)
    gt_boxes = gt_boxes[:, :4]
    gt_boxes, non_zeros = trim_padding_boxes(gt_boxes)
    gt_labels = tf.boolean_mask(gt_labels, non_zeros)
    meta_select_weight = tf.boolean_mask(meta_select_weight, non_zeros)

    def do_have_gt_boxes():
        cls_target = tf.zeros((0, num_classes + 1 + 1), dtype=tf.float32)
        regr_target = tf.zeros((0, 4 + 1 + 1), dtype=tf.float32)
        for level_id in range(len(strides)):
            level_meta_select_weight = meta_select_weight[:, level_id]

            fm_shape = fm_shapes[level_id]
            stride = strides[level_id]
            fh = fm_shape[0]
            fw = fm_shape[1]

            pos_x1, pos_y1, pos_x2, pos_y2 = shrink_and_project_boxes(gt_boxes, fw, fh, stride, shrink_ratio)

            def build_single_gt_box_sapd_target(args):
                pos_x1_ = args[0]
                pos_y1_ = args[1]
                pos_x2_ = args[2]
                pos_y2_ = args[3]
                gt_box = args[4]
                gt_label = args[5]
                level_box_meta_select_weight = args[6]

                level_pos_box_cls_target = tf.zeros((pos_y2_ - pos_y1_, pos_x2_ - pos_x1_, num_classes),
                                                    dtype=tf.float32)
                level_pos_box_gt_label_col = tf.ones((pos_y2_ - pos_y1_, pos_x2_ - pos_x1_, 1),
                                                     dtype=tf.float32)
                level_pos_box_cls_target = tf.concat((level_pos_box_cls_target[..., :gt_label],
                                                      level_pos_box_gt_label_col,
                                                      level_pos_box_cls_target[..., gt_label + 1:]), axis=-1)
                neg_top_bot = tf.stack((pos_y1_, fh - pos_y2_), axis=0)
                neg_lef_rit = tf.stack((pos_x1_, fw - pos_x2_), axis=0)
                neg_pad = tf.stack([neg_top_bot, neg_lef_rit], axis=0)
                level_box_cls_target = tf.pad(level_pos_box_cls_target,
                                              tf.concat((neg_pad, tf.constant([[0, 0]])), axis=0))
                pos_locs_x = tf.cast(tf.range(pos_x1_, pos_x2_), dtype=tf.float32)
                pos_locs_y = tf.cast(tf.range(pos_y1_, pos_y2_), dtype=tf.float32)
                pos_shift_x = (pos_locs_x + 0.5) * stride
                pos_shift_y = (pos_locs_y + 0.5) * stride
                pos_shift_xx, pos_shift_yy = tf.meshgrid(pos_shift_x, pos_shift_y)
                pos_shifts = tf.stack((pos_shift_xx, pos_shift_yy, pos_shift_xx, pos_shift_yy), axis=-1)
                dl = tf.maximum(pos_shifts[:, :, 0] - gt_box[0], 0)
                dt = tf.maximum(pos_shifts[:, :, 1] - gt_box[1], 0)
                dr = tf.maximum(gt_box[2] - pos_shifts[:, :, 2], 0)
                db = tf.maximum(gt_box[3] - pos_shifts[:, :, 3], 0)
                deltas = tf.stack((dl, dt, dr, db), axis=-1)
                level_box_regr_pos_target = deltas / 4.0 / stride
                level_pos_box_ap_weight = tf.minimum(dl, dr) * tf.minimum(dt, db) / tf.maximum(dl, dr) / tf.maximum(dt,
                                                                                                                    db)
                level_pos_box_soft_weight = level_pos_box_ap_weight * level_box_meta_select_weight
                level_box_soft_weight = tf.pad(level_pos_box_soft_weight, neg_pad, constant_values=1.)
                level_pos_box_regr_mask = tf.ones((pos_y2_ - pos_y1_, pos_x2_ - pos_x1_))
                level_box_regr_mask = tf.pad(level_pos_box_regr_mask, neg_pad)
                level_box_regr_target = tf.pad(level_box_regr_pos_target,
                                               tf.concat((neg_pad, tf.constant([[0, 0]])), axis=0))
                level_box_cls_target = tf.concat([level_box_cls_target, level_box_soft_weight[..., None],
                                                  level_box_regr_mask[..., None]], axis=-1)
                level_box_regr_target = tf.concat([level_box_regr_target, level_box_soft_weight[..., None],
                                                   level_box_regr_mask[..., None]], axis=-1)
                level_box_pos_area = (dl + dr) * (dt + db)
                level_box_area = tf.pad(level_box_pos_area, neg_pad, constant_values=1e7)
                return level_box_cls_target, level_box_regr_target, level_box_area

            level_cls_target, level_regr_target, level_area = tf.map_fn(
                build_single_gt_box_sapd_target,
                elems=[pos_x1, pos_y1, pos_x2, pos_y2, gt_boxes, gt_labels, level_meta_select_weight],
                dtype=(tf.float32, tf.float32, tf.float32)
            )
            level_min_area_box_indices = tf.argmin(level_area, axis=0, output_type=tf.int32)
            level_min_area_box_indices = tf.reshape(level_min_area_box_indices, (-1,))
            # (fw, )
            locs_x = tf.range(0, fw)
            # (fh, )
            locs_y = tf.range(0, fh)
            # (fh, fw), (fh, fw)
            locs_xx, locs_yy = tf.meshgrid(locs_x, locs_y)
            locs_xx = tf.reshape(locs_xx, (-1,))
            locs_yy = tf.reshape(locs_yy, (-1,))
            # (fh * fw, 3)
            level_indices = tf.stack((level_min_area_box_indices, locs_yy, locs_xx), axis=-1)
            level_cls_target = tf.gather_nd(level_cls_target, level_indices)
            level_regr_target = tf.gather_nd(level_regr_target, level_indices)

            cls_target = tf.concat([cls_target, level_cls_target], axis=0)
            regr_target = tf.concat([regr_target, level_regr_target], axis=0)
        return [cls_target, regr_target]

    def do_not_have_gt_boxes():
        fa = tf.reduce_prod(fm_shapes, axis=-1)
        fa_sum = tf.reduce_sum(fa)
        cls_target = tf.zeros((fa_sum, num_classes))
        regr_target = tf.zeros((fa_sum, 4))
        weight = tf.ones((fa_sum, 1))
        mask = tf.zeros((fa_sum, 1))
        cls_target = tf.concat([cls_target, weight, mask], axis=-1)
        regr_target = tf.concat([regr_target, weight, mask], axis=-1)
        return [cls_target, regr_target]

    cls_target, regr_target = tf.cond(
        tf.not_equal(tf.size(gt_boxes), 0),
        do_have_gt_boxes,
        do_not_have_gt_boxes
    )
    return [cls_target, regr_target]


class SAPDTarget(Layer):
    def __init__(self, num_classes, strides=(8, 16, 32, 64, 128), shrink_ratio=0.2, **kwargs):
        self.num_classes = num_classes
        self.strides = strides
        self.shrink_ratio = shrink_ratio
        super(SAPDTarget, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        fm_shapes = inputs[0][0]
        batch_gt_boxes = inputs[1]
        batch_meta_select_weight = inputs[2]

        def _build_sapd_target(args):
            gt_boxes = args[0]
            meta_select_weight = args[1]

            return build_sapd_target(
                gt_boxes,
                meta_select_weight,
                fm_shapes=fm_shapes,
                num_classes=self.num_classes,
                strides=self.strides,
                shrink_ratio=self.shrink_ratio,
            )

        outputs = tf.map_fn(
            _build_sapd_target,
            elems=[batch_gt_boxes, batch_meta_select_weight],
            dtype=[tf.float32, tf.float32],
        )
        return outputs

    def compute_output_shape(self, input_shape):
        batch_size = input_shape[0][0]
        return [[batch_size, None, self.num_classes + 1 + 1],
                [batch_size, None, 4 + 1 + 1]]

    def get_config(self):
        """
        Gets the configuration of this layer.

        Returns
            Dictionary containing the parameters of this layer.
        """
        config = super(SAPDTarget, self).get_config()
        config.update({'num_classes': self.num_classes})
        return config


class Locations(Layer):
    """
    Keras layer for generating anchors for a given shape.
    """

    def __init__(self, strides=(8, 16, 32, 64, 128), **kwargs):
        """
        Initializer for an Anchors layer.

        Args
            strides: The strides mapping to the feature maps.
        """
        self.strides = strides

        super(Locations, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        pyramid_features = inputs
        feature_shapes = [tf.shape(feature)[1:3] for feature in pyramid_features]
        locations_per_feature = []
        strides_per_feature = []
        for feature_shape, stride in zip(feature_shapes, self.strides):
            fh = feature_shape[0]
            fw = feature_shape[1]
            shifts_x = tf.cast(tf.range(0, fw * stride, delta=stride), dtype=tf.float32)
            shifts_y = tf.cast(tf.range(0, fh * stride, delta=stride), dtype=tf.float32)
            shift_x, shift_y = tf.meshgrid(shifts_x, shifts_y)
            # (h * w, )
            shift_x = tf.reshape(shift_x, (-1,))
            # (h * w, )
            shift_y = tf.reshape(shift_y, (-1,))
            locations = tf.stack((shift_x, shift_y), axis=1) + stride // 2
            locations_per_feature.append(locations)

            strides = tf.ones((fh, fw)) * stride
            strides = tf.reshape(strides, (-1,))
            strides_per_feature.append(strides)
        # (sum(h * w), 2)
        locations = tf.concat(locations_per_feature, axis=0)
        # (batch, sum(h * w), 2)
        locations = tf.tile(tf.expand_dims(locations, axis=0), (tf.shape(inputs[0])[0], 1, 1))
        strides = tf.concat(strides_per_feature, axis=0)
        strides = tf.tile(tf.expand_dims(strides, axis=0), (tf.shape(inputs[0])[0], 1))
        return [locations, strides]

    def compute_output_shape(self, input_shapes):
        feature_shapes = [feature_shape[1:3] for feature_shape in input_shapes]
        total = 1
        for feature_shape in feature_shapes:
            if None not in feature_shape:
                total = total * feature_shape[0] * feature_shape[1]
            else:
                return [[input_shapes[0][0], None, 2], [input_shapes[0][0], None]]
        return [[input_shapes[0][0], total, 2], [input_shapes[0][0], total]]

    def get_config(self):
        base_config = super(Locations, self).get_config()
        base_config.update({'strides': self.strides})
        return base_config


class RegressBoxes(Layer):
    """
    Keras layer for applying regression values to boxes.
    """

    def __init__(self, *args, **kwargs):
        """
        Initializer for the RegressBoxes layer.

        """
        super(RegressBoxes, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        locations, strides, regression = inputs
        x1 = locations[:, :, 0] - regression[:, :, 0] * 4.0 * strides[:, :]
        y1 = locations[:, :, 1] - regression[:, :, 1] * 4.0 * strides[:, :]
        x2 = locations[:, :, 0] + regression[:, :, 2] * 4.0 * strides[:, :]
        y2 = locations[:, :, 1] + regression[:, :, 3] * 4.0 * strides[:, :]
        bboxes = tf.stack([x1, y1, x2, y2], axis=-1)
        return bboxes

    def compute_output_shape(self, input_shape):
        return input_shape[2]

    def get_config(self):
        base_config = super(RegressBoxes, self).get_config()

        return base_config


class ClipBoxes(Layer):
    """
    Keras layer to clip box values to lie inside a given shape.
    """

    def call(self, inputs, **kwargs):
        image, boxes = inputs
        shape = tf.cast(tf.shape(image), tf.float32)
        height = shape[1]
        width = shape[2]
        x1 = tf.clip_by_value(boxes[:, :, 0], 0, width - 1)
        y1 = tf.clip_by_value(boxes[:, :, 1], 0, height - 1)
        x2 = tf.clip_by_value(boxes[:, :, 2], 0, width - 1)
        y2 = tf.clip_by_value(boxes[:, :, 3], 0, height - 1)

        return tf.stack([x1, y1, x2, y2], axis=2)

    def compute_output_shape(self, input_shape):
        return input_shape[1]
