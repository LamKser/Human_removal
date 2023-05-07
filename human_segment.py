from argparse import ArgumentParser

import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--weight', default='drafts/weights/best.pt', 
                        help='Yolov8 weight path')
    parser.add_argument('-iou', type=float, default=0.7,
                        help='IOU score')
    parser.add_argument('--conf', type=float, default=0.8,
                        help='Confidence')
    parser.add_argument('--image', required=True,
                        help='Image path')
    parser.add_argument('--save', action='store_true',
                        help='Save segmentation result')
    parser.add_argument('--save-crop', action='store_true',
                        help='Save crop segmentation result')
    parser.add_argument('--show', action='store_true',
                        help='Show segmentation result')
    args = parser.parse_args()

    img = cv2.imread(args.image)
    h, w = img.shape[:2]
    mask = np.zeros((h, w))

    model = YOLO(args.weight)
    results = model(img, 
                    conf=args.conf, iou=args.iou,
                    save=args.save, save_crop=args.save_crop)
    
    for result in results:
        mask_tmp = result.masks
        xy = mask_tmp.xy 
        for xy_ in xy:
            cv2.fillPoly(mask, [xy_.astype(int)], 255)
        
    if args.show:
        res_plot = results[0].plot()
        plt.imshow(res_plot[:, :, ::-1])
        plt.show()
