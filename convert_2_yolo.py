from argparse import ArgumentParser
import json
import os

from tqdm import tqdm
import numpy as np

import utils


def supervise_2_yolo(data_root, annotation_text_file):
    print(2345678)
    with open(annotation_text_file, 'w') as anno_text:
        for i in tqdm(range(1, 14)):
            num = f'ds{i}'
            path = f'{data_root}/{num}/ann'
            
            for root, _, ann_json in os.walk(path):
                for ann in ann_json:
                    with open(os.path.join(root, ann), 'r') as f:
                        meta = json.load(f)
                        height = meta['size']['height']
                        width = meta['size']['width']
                        objects = meta['objects']

                        for obj in objects:
                            if obj['geometryType'] == 'bitmap':
                                bitmap = obj['bitmap']
                                data  = bitmap['data']
                                origin = bitmap['origin']
                                
                                mask_tmp = utils.base64_2_mask(data)
                                h_tmp, w_tmp = mask_tmp.shape[:2]
                                xmin, ymin, xmax, ymax = utils.xywh_2_xyxy(origin, h_tmp, w_tmp)
                                mask = np.zeros((height, width))

                                mask[ymin:ymax, xmin:xmax] = mask_tmp
                                poly_points = utils.mask_2_polygons(mask)
                                
                            elif obj['geometryType'] == 'polygon':
                                poly_points = obj['points']['exterior']
                        
                            str_p = utils.point_2_str(poly_points, height, width)
                            anno_text.write(f'0\t{num}/img/{ann[:-5]}\t{str_p}\n')


def LVMHPv2_2_yolo(data_root, annotation_text_file):
     with open(annotation_text_file, 'w') as anno_text:
        for i in tqdm(range(1, 14)):
            num = f'ds{i}'
            path = f'{data_root}/{num}/ann'
            
            for root, _, ann_json in os.walk(path):
                for ann in ann_json:
                    with open(os.path.join(root, ann), 'r') as f:
                        meta = json.load(f)
                        height = meta['size']['height']
                        width = meta['size']['width']
                        objects = meta['objects']

                        for obj in objects:
                            if obj['geometryType'] == 'bitmap':
                                bitmap = obj['bitmap']
                                data  = bitmap['data']
                                origin = bitmap['origin']
                                
                                mask_tmp = utils.base64_2_mask(data)
                                h_tmp, w_tmp = mask_tmp.shape[:2]
                                xmin, ymin, xmax, ymax = utils.xywh_2_xyxy(origin, h_tmp, w_tmp)
                                mask = np.zeros((height, width))

                                mask[ymin:ymax, xmin:xmax] = mask_tmp
                                poly_points = utils.mask_2_polygons(mask)
                                
                            elif obj['geometryType'] == 'polygon':
                                poly_points = obj['points']['exterior']
                        
                            str_p = utils.point_2_str(poly_points, height, width)
                            anno_text.write(f'0\t{num}/img/{ann[:-5]}\t{str_p}\n')



if __name__ == '__main__':
    data = {1: ['supervisely', supervise_2_yolo],
            2: ['LV MHP v2', LVMHPv2_2_yolo]}

    print('============Data============')
    for k, v in data.items():
        print(f'{k}. {v[0]}')

    parser = ArgumentParser()
    parser.add_argument('--type', type=int, choices=list(data.keys()), required=True,
                        help='Choose dataset')
    parser.add_argument('--data-root', type=str, required=True,
                        help='Path of dataset')
    parser.add_argument('--text', type=str, default='annotations',
                        help='Annotation text file (.txt)')
    
    args = parser.parse_args()

    if 'txt' not in args.text:
        args.text = args.text + '.txt'

    data[args.type][-1](args.data_root, args.text)