# object-detection-level2-cv-13

# 프로젝트 개요

 재활용 쓰레기의 분리배출을 통해 각종 쓰레기 처리 문제를 줄이고 환경 부담을 줄일 수 있습니다. 재활용 쓰레기의 정확한 분리배출을 돕거나 어린이의 분리배출 교육 등 여러방면에서 사용될 수 있는 Object Detection 모델을 만들었습니다.

![7645ad37-9853-4a85-b0a8-f0f151ef05be](https://user-images.githubusercontent.com/47216338/137615872-208f08db-55a8-4100-a65b-075cb035238c.png)

# 디렉토리 구조

```
object-detection-level2-cv-13/
|___ efficientdet
|___ experiments 
|___ mmdetection
|       |___ custom_configs
|___ yolov5
```

- efficientdet : EfficientDet 모델 training 및 inference
- experiments : EDA, custom augmentation, validation set 생성 등 실험에 관해 작성한 코드
- mmdetection/custom_configs : mmdetection을 이용한 실험에서 사용한 config 파일
- yolov5 : YOLOv5 모델 training 및 inference

# 주요 명령어

- Efficientdet
    - Train
        
        ```
        python ./efficientdet/train.py
        ```
        
    - Inference
        
        ```
        python ./efficientdet/inference.py
        ```
        
- MMdetection
    - Train
        
        ```
        python ./mmdetction/tools/train.py [config 파일 경로]
        ```
        
    - Inference
        
        ```
        python ./mmdetection/tools/SDJ_test.py --config [config 파일 경로] --checkpoint [checkpoint 파일 경로]
        ```
        
- Yolov5
    - Train
        
        ```
        python ./yolov5/train.py --data coco.yaml --cfg yolov5s.yaml --weights '' --batch-size 64
                                                 yolov5m                                40
                                                 yolov5l                                24
                                                 yolov5x                                16
        ```
        
    - Inference
        
        ```
        $ python ./yolov5/detect.py --source 0  # webcam
                                    file.jpg  # image 
                                    file.mp4  # video
                                    path/  # directory
                                    path/*.jpg  # glob
                                    'https://youtu.be/NUsoVlDFqZg'  # YouTube
                                    'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream
        ```
        
        ```
        python ./yolov5/submission.py --submission_path [csv 생성 경로] --result_path [detcect.py 실행 결과 파일 경로]
        ```
        
        - detect.py를 실행한 결과를 이용해 inference 결과를 얻고, submission.py를 통해 csv 형태 변환합니다.
- 앙상블
    
    ```
    python ./mmdetection/ensemble_inference.py ensemble_inference_cfg.json
    ```
    
    - ensemble_inference_cfg.json 파일을 통해 앙상블하고자 하는 csv파일들과 모드를 설정하고 ensemble_inference..py 파일을 실행합니다.

# 참조 자료

- mmdetection
    - [https://github.com/open-mmlab/mmdetection](https://github.com/open-mmlab/mmdetection)
    - [https://mmdetection.readthedocs.io](https://mmdetection.readthedocs.io/en/latest/)
- yolov5
    - [https://github.com/ultralytics/yolov5](https://github.com/ultralytics/yolov5)

