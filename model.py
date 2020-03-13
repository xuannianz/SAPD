from functools import reduce

# from keras import layers
# from keras import initializers
# from keras import models
# from keras import losses
# from keras_ import EfficientNetB0, EfficientNetB1, EfficientNetB2
# from keras_ import EfficientNetB3, EfficientNetB4, EfficientNetB5, EfficientNetB6

from tensorflow.keras import layers
from tensorflow.keras import initializers
from tensorflow.keras import models
from tensorflow.keras import losses
from tfkeras import EfficientNetB0, EfficientNetB1, EfficientNetB2
from tfkeras import EfficientNetB3, EfficientNetB4, EfficientNetB5, EfficientNetB6

from layers import BatchNormalization, FilterDetections, ClipBoxes
from losses import focal_with_weight_and_mask, iou_with_weight_and_mask
from sapd_layers import MetaSelectTarget, MetaSelectInput, SAPDTarget, Locations, RegressBoxes, MetaSelectWeight
from initializers import PriorProbability, FixedValueBiasInitializer

w_bifpns = [64, 88, 112, 160, 224, 288, 384]
image_sizes = [512, 640, 768, 896, 1024, 1280, 1408]
backbones = [EfficientNetB0, EfficientNetB1, EfficientNetB2,
             EfficientNetB3, EfficientNetB4, EfficientNetB5, EfficientNetB6]


def DepthwiseSeparableConvBlock(num_channels, kernel_size, strides, name, freeze_bn=False):
    f1 = layers.SeparableConv2D(num_channels, kernel_size=kernel_size, strides=strides, padding='same',
                                use_bias=False, name='{}_dconv'.format(name))
    f2 = BatchNormalization(freeze=freeze_bn, name='{}_bn'.format(name))
    f3 = layers.ReLU(name='{}_relu'.format(name))
    return reduce(lambda f, g: lambda *args, **kwargs: g(f(*args, **kwargs)), (f1, f2, f3))


def ConvBlock(num_channels, kernel_size, strides, name, freeze_bn=False):
    f1 = layers.Conv2D(num_channels, kernel_size=kernel_size, strides=strides, padding='same',
                       use_bias=False, name='{}_conv'.format(name))
    f2 = BatchNormalization(freeze=freeze_bn, name='{}_bn'.format(name))
    f3 = layers.ReLU(name='{}_relu'.format(name))
    return reduce(lambda f, g: lambda *args, **kwargs: g(f(*args, **kwargs)), (f1, f2, f3))


def build_BiFPN(features, num_channels, id, freeze_bn=False):
    if id == 0:
        _, _, C3, C4, C5 = features
        P3_in = ConvBlock(num_channels, kernel_size=1, strides=1, freeze_bn=freeze_bn, name='BiFPN_{}_P3'.format(id))(
            C3)
        P4_in = ConvBlock(num_channels, kernel_size=1, strides=1, freeze_bn=freeze_bn, name='BiFPN_{}_P4'.format(id))(
            C4)
        P5_in = ConvBlock(num_channels, kernel_size=1, strides=1, freeze_bn=freeze_bn, name='BiFPN_{}_P5'.format(id))(
            C5)
        P6_in = ConvBlock(num_channels, kernel_size=3, strides=2, freeze_bn=freeze_bn, name='BiFPN_{}_P6'.format(id))(
            C5)
        P7_in = ConvBlock(num_channels, kernel_size=3, strides=2, freeze_bn=freeze_bn, name='BiFPN_{}_P7'.format(id))(
            P6_in)
    else:
        P3_in, P4_in, P5_in, P6_in, P7_in = features
        P3_in = ConvBlock(num_channels, kernel_size=1, strides=1, freeze_bn=freeze_bn, name='BiFPN_{}_P3'.format(id))(
            P3_in)
        P4_in = ConvBlock(num_channels, kernel_size=1, strides=1, freeze_bn=freeze_bn, name='BiFPN_{}_P4'.format(id))(
            P4_in)
        P5_in = ConvBlock(num_channels, kernel_size=1, strides=1, freeze_bn=freeze_bn, name='BiFPN_{}_P5'.format(id))(
            P5_in)
        P6_in = ConvBlock(num_channels, kernel_size=1, strides=1, freeze_bn=freeze_bn, name='BiFPN_{}_P6'.format(id))(
            P6_in)
        P7_in = ConvBlock(num_channels, kernel_size=1, strides=1, freeze_bn=freeze_bn, name='BiFPN_{}_P7'.format(id))(
            P7_in)

    # upsample
    P7_U = layers.UpSampling2D()(P7_in)
    P6_td = layers.Add()([P7_U, P6_in])
    P6_td = DepthwiseSeparableConvBlock(num_channels, kernel_size=3, strides=1, freeze_bn=freeze_bn,
                                        name='BiFPN_{}_U_P6'.format(id))(P6_td)
    P6_U = layers.UpSampling2D()(P6_td)
    P5_td = layers.Add()([P6_U, P5_in])
    P5_td = DepthwiseSeparableConvBlock(num_channels, kernel_size=3, strides=1, freeze_bn=freeze_bn,
                                        name='BiFPN_{}_U_P5'.format(id))(P5_td)
    P5_U = layers.UpSampling2D()(P5_td)
    P4_td = layers.Add()([P5_U, P4_in])
    P4_td = DepthwiseSeparableConvBlock(num_channels, kernel_size=3, strides=1, freeze_bn=freeze_bn,
                                        name='BiFPN_{}_U_P4'.format(id))(P4_td)
    P4_U = layers.UpSampling2D()(P4_td)
    P3_out = layers.Add()([P4_U, P3_in])
    P3_out = DepthwiseSeparableConvBlock(num_channels, kernel_size=3, strides=1, freeze_bn=freeze_bn,
                                         name='BiFPN_{}_U_P3'.format(id))(P3_out)
    # downsample
    P3_D = layers.MaxPooling2D(strides=(2, 2))(P3_out)
    P4_out = layers.Add()([P3_D, P4_td, P4_in])
    P4_out = DepthwiseSeparableConvBlock(num_channels, kernel_size=3, strides=1, freeze_bn=freeze_bn,
                                         name='BiFPN_{}_D_P4'.format(id))(P4_out)
    P4_D = layers.MaxPooling2D(strides=(2, 2))(P4_out)
    P5_out = layers.Add()([P4_D, P5_td, P5_in])
    P5_out = DepthwiseSeparableConvBlock(num_channels, kernel_size=3, strides=1, freeze_bn=freeze_bn,
                                         name='BiFPN_{}_D_P5'.format(id))(P5_out)
    P5_D = layers.MaxPooling2D(strides=(2, 2))(P5_out)
    P6_out = layers.Add()([P5_D, P6_td, P6_in])
    P6_out = DepthwiseSeparableConvBlock(num_channels, kernel_size=3, strides=1, freeze_bn=freeze_bn,
                                         name='BiFPN_{}_D_P6'.format(id))(P6_out)
    P6_D = layers.MaxPooling2D(strides=(2, 2))(P6_out)
    P7_out = layers.Add()([P6_D, P7_in])
    P7_out = DepthwiseSeparableConvBlock(num_channels, kernel_size=3, strides=1, freeze_bn=freeze_bn,
                                         name='BiFPN_{}_D_P7'.format(id))(P7_out)

    return P3_out, P4_out, P5_out, P6_out, P7_out


def build_regress_head(width, depth):
    options = {
        'kernel_size': 3,
        'strides': 1,
        'padding': 'same',
        'activation': 'relu',
        'kernel_initializer': initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None),
    }

    inputs = layers.Input(shape=(None, None, width))
    outputs = inputs
    for i in range(depth):
        outputs = layers.Conv2D(
            filters=width,
            bias_initializer='zeros',
            **options
        )(outputs)

    outputs = layers.Conv2D(4, bias_initializer=FixedValueBiasInitializer(0.1), **options)(outputs)
    outputs = layers.Reshape((-1, 4), name='class_head_reshape')(outputs)

    return models.Model(inputs=inputs, outputs=outputs, name='box_head')


def build_class_head(width, depth, num_classes=20):
    options = {
        'kernel_size': 3,
        'strides': 1,
        'padding': 'same',
        'kernel_initializer': initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None),
    }

    inputs = layers.Input(shape=(None, None, width))
    outputs = inputs
    for i in range(depth):
        outputs = layers.Conv2D(
            filters=width,
            activation='relu',
            bias_initializer='zeros',
            **options
        )(outputs)

    outputs = layers.Conv2D(
        filters=num_classes,
        bias_initializer=PriorProbability(probability=0.01),
        activation='sigmoid',
        **options
    )(outputs)
    outputs = layers.Reshape((-1, num_classes), name='class_head_reshape')(outputs)
    return models.Model(inputs=inputs, outputs=outputs, name='class_head')


def build_meta_select_net(width=256, depth=3, pool_size=7, num_pyramid_levels=5):
    options = {
        'kernel_size': 3,
        'strides': 1,
        'kernel_initializer': initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None),
        'bias_initializer': 'zeros'
    }
    inputs = layers.Input(shape=(pool_size, pool_size, width * num_pyramid_levels))
    outputs = inputs
    for i in range(depth):
        outputs = layers.Conv2D(filters=width, activation='relu', **options)(outputs)
    outputs = layers.Flatten()(outputs)
    outputs = layers.Dense(num_pyramid_levels,
                           kernel_initializer=initializers.RandomNormal(mean=0., stddev=0.01, seed=None),
                           bias_initializer='zeros',
                           activation='softmax'
                           )(outputs)
    return models.Model(inputs=inputs, outputs=outputs, name='meta_select_net')


def sapd(phi, soft_select=False, num_classes=20, freeze_bn=False, max_gt_boxes=100,
         batch_size=32,
         score_threshold=0.01,
         ):
    assert phi in range(7)
    image_size = image_sizes[phi]
    input_shape = (image_size, image_size, 3)
    # input_shape = (None, None, 3)
    image_input = layers.Input(input_shape)
    gt_boxes_input = layers.Input((max_gt_boxes, 5))
    num_gt_boxes_input = layers.Input((1,), dtype='int32')
    fm_shapes_input = layers.Input((5, 2), dtype='int32')

    backbone_cls = backbones[phi]
    # (C1, C2, C3, C4, C5)
    features = backbone_cls(input_tensor=image_input, freeze_bn=freeze_bn)
    w_bifpn = w_bifpns[phi]
    d_bifpn = 2 + phi
    w_head = w_bifpn
    d_head = 3 + int(phi / 3)
    for i in range(d_bifpn):
        features = build_BiFPN(features, w_bifpn, i, freeze_bn=freeze_bn)
    regr_head = build_regress_head(w_head, d_head)
    cls_head = build_class_head(w_head, d_head, num_classes=num_classes)
    pyramid_features = features
    fpn_width = w_head
    cls_pred = [cls_head(feature) for feature in pyramid_features]
    cls_pred = layers.Concatenate(axis=1, name='classification')(cls_pred)
    regr_pred = [regr_head(feature) for feature in pyramid_features]
    regr_pred = layers.Concatenate(axis=1, name='regression')(regr_pred)

    # meta select net
    meta_select_net = build_meta_select_net(width=fpn_width)
    meta_select_input, gt_boxes_batch_ids = MetaSelectInput()([gt_boxes_input, *pyramid_features])
    meta_select_pred = meta_select_net(meta_select_input)
    meta_select_target = MetaSelectTarget()([cls_pred, regr_pred, fm_shapes_input, gt_boxes_input])
    # # lambda == 0.1 in paper
    meta_select_loss = layers.Lambda(lambda x: 0.1 * losses.sparse_categorical_crossentropy(x[0], x[1]),
                                     output_shape=(1,),
                                     name="meta_select_loss")([meta_select_target, meta_select_pred])

    if soft_select:
        meta_select_weight = MetaSelectWeight(max_gt_boxes=max_gt_boxes,
                                              soft_select=soft_select,
                                              batch_size=batch_size,
                                              )([meta_select_pred, gt_boxes_batch_ids, num_gt_boxes_input])
    else:
        meta_select_weight = MetaSelectWeight(max_gt_boxes=max_gt_boxes,
                                              soft_select=soft_select,
                                              batch_size=batch_size,
                                              )([meta_select_target, gt_boxes_batch_ids, num_gt_boxes_input])

    cls_target, regr_target = SAPDTarget(num_classes=num_classes)([fm_shapes_input,
                                                                   gt_boxes_input,
                                                                   meta_select_weight])

    focal_loss = focal_with_weight_and_mask()
    iou_loss = iou_with_weight_and_mask()
    cls_loss = layers.Lambda(focal_loss,
                             output_shape=(1,),
                             name="cls_loss")([cls_target, cls_pred])
    regr_loss = layers.Lambda(iou_loss,
                              output_shape=(1,),
                              name="regr_loss")([regr_target, regr_pred])

    model = models.Model(inputs=[image_input, gt_boxes_input, num_gt_boxes_input, fm_shapes_input],
                         outputs=[cls_loss, regr_loss, meta_select_loss, cls_pred, regr_pred, cls_target,
                                  regr_target],
                         name='sapd')

    locations, strides = Locations()(pyramid_features)

    # apply predicted regression to anchors
    boxes = RegressBoxes(name='boxes')([locations, strides, regr_pred])
    boxes = ClipBoxes(name='clipped_boxes')([image_input, boxes])

    # filter detections (apply NMS / score threshold / select top-k)
    detections = FilterDetections(
        name='filtered_detections',
        score_threshold=score_threshold
    )([boxes, cls_pred])

    prediction_model = models.Model(inputs=[image_input], outputs=detections, name='sapd_p')

    return model, prediction_model


if __name__ == '__main__':
    model, _ = sapd(0)
    model.summary()
