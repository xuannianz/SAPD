from model import sapd
import cv2
import os
import numpy as np
import time
import glob

from utils import preprocess_image
from utils.draw_boxes import draw_boxes
from utils.post_process_boxes import post_process_boxes


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    phi = 0
    feature_fusion = False
    model_path = 'checkpoints/pascal_16_0.2373_0.5481_0.7820_0.7850.h5'
    image_sizes = (512, 640, 768, 896, 1024, 1280, 1408)
    image_size = image_sizes[phi]
    classes = [
        'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair',
        'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
        'tvmonitor',
    ]
    num_classes = len(classes)
    score_threshold = 0.3
    colors = [np.random.randint(0, 256, 3).tolist() for i in range(num_classes)]
    model, prediction_model = sapd(phi=phi,
                                   num_classes=num_classes,
                                   score_threshold=score_threshold,
                                   feature_fusion=feature_fusion
                                   )
    prediction_model.load_weights(model_path, by_name=True)

    for image_path in glob.glob('datasets/VOC2007/JPEGImages/*.jpg'):
        image = cv2.imread(image_path)
        src_image = image.copy()
        image = image[:, :, ::-1]
        h, w = image.shape[:2]

        image, scale, offset_h, offset_w = preprocess_image(image, image_size=image_size)
        # run network
        start = time.time()
        boxes, scores, labels = prediction_model.predict_on_batch([np.expand_dims(image, axis=0)])
        boxes, scores, labels = np.squeeze(boxes), np.squeeze(scores), np.squeeze(labels)
        print(time.time() - start)
        boxes = post_process_boxes(boxes=boxes,
                                   scale=scale,
                                   offset_h=offset_h,
                                   offset_w=offset_w,
                                   height=h,
                                   width=w)

        # select indices which have a score above the threshold
        indices = np.where(scores[:] > score_threshold)[0]

        # select those detections
        boxes = boxes[indices]
        labels = labels[indices]

        draw_boxes(src_image, boxes, scores, labels, colors, classes)

        cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        cv2.imshow('image', src_image)
        key = cv2.waitKey(0)
        if chr(key) == 'y':
            import os.path as osp
            image_filename = osp.split(image_path)[-1]
            cv2.imwrite(f'test/{image_filename}', src_image)


if __name__ == '__main__':
    main()
