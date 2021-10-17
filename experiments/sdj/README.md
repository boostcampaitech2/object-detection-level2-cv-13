# 파일 설명

경진대회를 진행하며 추가적으로 사용하였던 파일들과 그에 대한 설명입니다.



#### split_to_json.py

class를 stratified하게 나누고 각 image를 group으로 설정하여 annotation 정보를 새롭게 저장하는 .py 실행 파일입니다.

이 과정에 outlier에 대한 setting을 설정하여 outlier를 제거할 수 있지만 최종적으로 사용하지 않았기에 기본 실행 방법만 설명합니다.

**실행 명령어**

```python
python split_to_json.py --split_num [split 수] --seed [seed number] --json_save_folder [json이 저장될 폴더]
```



#### EDA.ipynb

전반적인 데이터 탐색과정이 담긴 주피터 노트북 파일입니다.



#### mmdetection_aug_check.ipynb

MMdetection을 이용한 augmentation이 어떻게 이미지에 적용되는지 확인하기 위한 주피터 노트북 파일입니다.
