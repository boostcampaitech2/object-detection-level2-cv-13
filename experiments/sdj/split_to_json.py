from rm_outlier import *
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from collections import Counter, defaultdict
from pycocotools.coco import COCO
import json
import pandas as pd
import numpy as np
import argparse 
import time
import os

# 실행방법 python split_to_json.py

#df_annotations.to_dict(


if __name__=="__main__":
    start_time = time.time()

    # parser.add_argument('--split_fold', type = int, help='validation split fold',default=0)
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--split_num', type = str, 
                        help='fold split number',default=5)
    parser.add_argument('--seed', type = str, 
                        help='fold split seed',default=42)
    parser.add_argument('--json_save_folder', type = str, 
                        help='folder_directory to save new json file',default='/opt/ml/detection/dataset/trn_val_split_json')
    parser.add_argument('--check_outlier',type =bool,  default = False, help = "check whether remove outlier or not")
    parser.add_argument('--rm_bbox',type =int,  default = 40, 
        help = "the number of annotations that a image is defined as outlier")
    parser.add_argument('--rm_wh', type = list, default = [10,10],
        help = "Width and Height of annotations to be defined as outlier")
    args = parser.parse_args() 

    
    # train(원본 4,883) 파일을 json과 coco type으로 불러오기
    with open("/opt/ml/detection/dataset/train.json") as f:
        train = json.load(f)
    coco = COCO('/opt/ml/detection/dataset/train.json') 


    # image category와 넓이에 맞게 split 하는 부분
    df_images = pd.json_normalize(train['images'])
    df_annotations = pd.json_normalize(train['annotations'])

    df_annotations['area_cut'] = pd.cut(df_annotations['area'],bins= [0,5000,20000,60000,200000,1050000]) 
    df_annotations['area_cut'] = df_annotations['area_cut'].apply(lambda x: int(np.log(x.right)))
    df_annotations['category_area'] = df_annotations['category_id'].astype(str) + "_" + df_annotations['area_cut'].astype(str)
    
    skf = StratifiedGroupKFold(n_splits = args.split_num, random_state = args.seed, shuffle = True)
    folds = skf.split(df_annotations['id'], df_annotations['category_area'], df_annotations['image_id'])
    for fold, (trn_idx, val_idx) in enumerate(folds):

        print(f"{fold} spliting...")
        # 원본 train json 이용하여 image, annotations 바꿔준다. 
        trn_json = train.copy()
        val_json = train.copy()
        trn_json['images'] = df_to_formatted_json(df_images.loc[df_annotations.loc[trn_idx,'image_id'].unique()]) 
        val_json['images'] = df_to_formatted_json(df_images.loc[df_annotations.loc[val_idx,'image_id'].unique()])

        trn_json['annotations'] = df_to_formatted_json(df_annotations.loc[trn_idx])
        val_json['annotations'] = df_to_formatted_json(df_annotations.loc[val_idx])


        # image id와 annotation id를 연속적이게 바꿔주는 부분
        # train
        start_ann_idx = 0
        trn_img_df = pd.json_normalize(trn_json['images'])
        for i, img_info in enumerate(trn_json['images']):
            image_id = trn_img_df['id'].iloc[i]
            ann_ids = coco.getAnnIds(imgIds=image_id)

            for ann_idx in range(start_ann_idx, start_ann_idx+len(ann_ids)):
                trn_json['annotations'][ann_idx]['image_id'] = i # annotation의 image_id 바꾼다.
                trn_json['annotations'][ann_idx]['id'] = ann_idx # annotations id 바꾼다.
            start_ann_idx = ann_idx + 1
            trn_json['images'][i]['id'] = i # image의 id 바꾼다.

        # valid
        start_ann_idx = 0
        val_img_df = pd.json_normalize(val_json['images'])
        for i, img_info in enumerate(val_json['images']):
            image_id = val_img_df['id'].iloc[i] # # 이미지 불러와서  
            ann_ids = coco.getAnnIds(imgIds=image_id) # 어노테이션 아디들 불러오고

            for ann_idx in range(start_ann_idx, start_ann_idx+len(ann_ids)):
                val_json['annotations'][ann_idx]['image_id'] = i
                val_json['annotations'][ann_idx]['id'] = ann_idx 
            start_ann_idx = ann_idx + 1
            val_json['images'][i]['id'] = i

        # folder 생성
        try: 
            if not os.path.exists(args.json_save_folder): 
                os.makedirs(args.json_save_folder) 
        except OSError: 
            print("Error: Failed to create the directory.")

        if args.check_outlier:
            trn_json = rm_outlier(trn_json, args.rm_bbox, args.rm_wh)
        #    val_json = rm_outlier(val_json, args.rm_bbox, args.rm_wh)

        with open(args.json_save_folder + f'/train_split_{fold}.json', 'w') as fp:
            json.dump(trn_json, fp)

        with open(args.json_save_folder + f'/valid_split_{fold}.json', 'w') as fp:
            json.dump(val_json, fp)

        print(f"{fold} end!")
    print(f"json files saved at {args.json_save_folder}")
    print(f"---{np.round(time.time()-start_time, 2)}s seconds---")