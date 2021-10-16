"""
detect.py의 결과물을 원하는 submission 형태로 변환
"""

import pandas as pd
import os, glob
import argparse

def main(args):
    """
    args에서 모델 결과 경로와 파일 저장 경로를 읽어와서 submission file을 생성함

    - Args
        args: ArgumentParser (모델 결과 경로와 submission이 저장될 경로를 설정)
    """

    # submission file
    preds = []
    names = []
    for res in sorted(glob.glob(args.result_path+"*.txt")):
        with open(res, "r") as f:
            preds.append(f.read().rstrip())
        names.append("test/"+res.split("/")[-1].replace("txt","jpg"))
    
    submission = pd.DataFrame()
    submission['PredictionString'] = preds
    submission['image_id'] = names

    submission.to_csv(f"{args.submission_path}/submission_yolov5.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse the path of a submission file")
    parser.add_argument("--submission_path", type=str, default="./", help="Path to save a submission file. (default: Current Working Directory")
    parser.add_argument("--result_path", type=str, default="./runs/detect/", help="Path that prediction results exist. (default: './runs/detect/'")
    args = parser.parse_args()
    main(args)