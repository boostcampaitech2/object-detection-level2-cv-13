import pandas as pd
import numpy as np
import os

from ensemble_boxes import *
import argparse
import json
from tqdm import tqdm

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('cfg', type=str)
    args = parser.parse_args()
    
    return args

def extracting_info(csvs, img_size):
    '''
    csv 파일의 정보를 추출하여 ensemble_boxes의 입력 형태로 변환
    (box, score, label)
    '''
    box_info = []
    score_info = []
    label_info = []

    for csv in tqdm(csvs):
        result = pd.read_csv(csv)
        
        box_info_per_csv = []
        score_info_per_csv = []
        label_info_per_csv = []
        for i, prediction in enumerate(result["PredictionString"]):
            box_info_per_img = []
            score_info_per_img = []
            label_info_per_img = []

            # 모델이 박스를 예측하지 않은 경우
            if isinstance(prediction, float):
                box_info_per_csv.append(box_info_per_img)
                score_info_per_csv.append(score_info_per_img)
                label_info_per_csv.append(label_info_per_img)

                continue

            prediction = prediction.split()
            
            tmp_box = []
            for i, pred in enumerate(prediction):
                if i % 6 == 0:
                    label_info_per_img.append(int(pred))
                elif i % 6 == 1:
                    score_info_per_img.append(float(pred))
                else:
                    if float(pred) / img_size > 1:
                        tmp_box.append(1.0)
                    else:
                        # 박스의 좌표를 이미지 에 대한 값으로 정규화 (0~1)
                        tmp_box.append(float(pred) / img_size)

                    if len(tmp_box) == 4:
                        box_info_per_img.append(tmp_box)
                        tmp_box = []

            box_info_per_csv.append(box_info_per_img)
            score_info_per_csv.append(score_info_per_img)
            label_info_per_csv.append(label_info_per_img)
        
        box_info.append(box_info_per_csv)
        score_info.append(score_info_per_csv)
        label_info.append(label_info_per_csv)

    return box_info, score_info, label_info

def make_submmission(save_path, result_box_info, result_score_info, result_label_info):
    '''
    최종 결과를 제출 양식에 맞게 csv 파일로 생성
    '''
    prediction_strings = []
    file_names = []
    for i, (box, score, label) in enumerate(zip(result_box_info, result_score_info, result_label_info)):
        file_names.append("test/" + str(i).zfill(4) + ".jpg")
        prediction = ''

        for b, s, l in zip(box, score, label):
            x_min = b[0]
            y_min = b[1]
            x_max = b[2]
            y_max = b[3]

            prediction += str(int(l)) + ' ' + str(s) + ' ' + str(x_min) + ' ' + str(y_min) + ' ' + str(x_max) + ' ' + str(y_max) + ' '
        
        prediction_strings.append(prediction)


    submission = pd.DataFrame()
    submission['PredictionString'] = prediction_strings
    submission['image_id'] = file_names
    submission.to_csv(save_path, index=None)

def main():
    args = arg_parse()

    with open(args.cfg, 'r') as f:
        cfgs = json.load(f)

    csvs = cfgs['csvs']
    mode = cfgs['ensemble_mode']
    save_path = cfgs['save_path']
    weights = cfgs['weights']
    iou_thr = cfgs['iou_thr']
    img_size = cfgs['img_size']

    if weights == 'None':
        weights = None

    num_models = len(csvs)
    print(csvs)

    box_info, score_info, label_info = extracting_info(csvs, img_size)
    
    if box_info:
        num_imgs = len(box_info[0])
    else:
        num_imgs = 0

    result_box_info = []
    result_score_info = []
    result_label_info = []

    for img_i in tqdm(range(0, num_imgs)):
        boxes_list = []
        scores_list = []
        labels_list = []
        
        for model_i in range(0, num_models):
            boxes_list.append(box_info[model_i][img_i])
            scores_list.append(score_info[model_i][img_i])
            labels_list.append(label_info[model_i][img_i])

        # 앙상블 mode 선택
        if mode == 'nms':
            boxes, scores, labels = nms(boxes_list, scores_list, labels_list, weights=weights, iou_thr=iou_thr)
        elif mode == 'snms':
            sigma = cfgs['sigma']
            skip_box_thr = cfgs['skip_box_thr']
            boxes, scores, labels = soft_nms(boxes_list, scores_list, labels_list, weights=weights, iou_thr=iou_thr, sigma=sigma, thresh=skip_box_thr)
        elif mode == 'nmw':
            skip_box_thr = cfgs['skip_box_thr']
            boxes, scores, labels = non_maximum_weighted(boxes_list, scores_list, labels_list, weights=weights, iou_thr=iou_thr, skip_box_thr=skip_box_thr)
        elif mode == 'wbf':
            skip_box_thr = cfgs['skip_box_thr']
            boxes, scores, labels = weighted_boxes_fusion(boxes_list, scores_list, labels_list, weights=weights, iou_thr=iou_thr, skip_box_thr=skip_box_thr)

        # 이미지 크기에 대한 값으로 정규화했던 값을 좌표값으로 변환 
        for b in boxes:
            for i in range(4):
                b[i] = b[i]*img_size

        result_box_info.append(boxes)
        result_score_info.append(scores)
        result_label_info.append(labels)

    make_submmission(save_path, result_box_info, result_score_info, result_label_info)


if __name__ == '__main__':
    main()
    print("done")
