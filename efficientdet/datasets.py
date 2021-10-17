from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import Dataset
from pycocotools.coco import COCO
import pandas as pd
import numpy as np
import random
import torch
import json
import cv2
import os


class TrainDataset(Dataset):

    def __init__(self, annotation, data_dir, mode, cutmix = True, mixup = True, fold = 0, k = 5, random_state = 923, transforms=None):
        
        """ Trash Object Detection Train Dataset
        Args:
            annotation : annotation directory
            data_dir : data_dir directory
            mode : "train" when you want to train, "validation" when you want to evaluate
            cutmix : True when you want to use cutmix(mosaic)
            mixup : True when you want to use mixup
            fold : the order of fold to be learned
            k : how many folds going to devided
            random_state : random state of kfold
            transforms : transforms to be applied to the image
        """
        
        super().__init__()
        self.data_dir = data_dir
        self.coco = COCO(annotation)
        self.mode = mode
        self.cutmix = cutmix
        self.mixup = mixup
        self.predictions = {
            "images": self.coco.dataset["images"].copy(),
            "categories": self.coco.dataset["categories"].copy(),
            "annotations": None
        }
        self.transforms = transforms

        with open(annotation) as f:
            train = json.load(f)

        df_images = pd.json_normalize(train['images'])
        df_annotations = pd.json_normalize(train['annotations'])
        train_df = df_images.set_index('id').join(df_annotations.set_index('image_id')).set_index('id')
        category_map = {x['id']: x['name'] + "("+str(x['id']) + ")" for x in train['categories']}
        train_df['category_name'] = train_df['category_id'].map(category_map)
        train_df['image_id'] = train_df['file_name'].apply(lambda x: x.split('/')[1].split('.')[0])
        train_df.reset_index(inplace = True)
        train_df['area_recat'] = train_df['area'].apply(self._area_func)
        train_df['area_cat'] = train_df['category_id'].astype(str) + train_df['area_recat'].astype(str)
        
        skf = StratifiedGroupKFold(n_splits = k, random_state = random_state, shuffle = True)
        for i, (train_idx, val_idx) in enumerate(skf.split(train_df['id'], train_df['area_cat'], train_df['image_id'])):
            if i == fold:
                break
        
        if self.mode == "train":
            self.img_idx = train_df.iloc[train_idx]['image_id'].astype(int).unique()
        if self.mode == "validation":
            self.img_idx = train_df.iloc[val_idx]['image_id'].astype(int).unique()


    def __len__(self) -> int:

        return len(self.img_idx)
        

    def __getitem__(self, index: int):
        
        random_number = random.random()
        if self.mode == "validation":
            image, boxes, labels, image_id = self.load_image_boxes_labels(index)
        elif self.mode == "train":
            if self.cutmix == False and self.mixup == False:
                image, boxes, labels, image_id = self.load_image_boxes_labels(index)
            elif self.cutmix == True and self.mixup == False:
                if random_number > 0.5:
                    image, boxes, labels, image_id = self.load_image_boxes_labels(index)
                else:
                    image, boxes, labels, image_id = self.load_cutmix(index)
            elif self.cutmix == False and self.mixup == True:
                if random_number > 0.5:
                    image, boxes, labels, image_id = self.load_image_boxes_labels(index)
                else:
                    image, boxes, labels, image_id = self.load_mixup(index)
            elif self.cutmix == True and self.mixup == True:
                if random_number > 0.5:
                    image, boxes, labels, image_id = self.load_image_boxes_labels(index)
                elif random_number > 0.25:
                    image, boxes, labels, image_id = self.load_cutmix(index)
                else:
                    image, boxes, labels, image_id = self.load_mixup(index)

        target = {'boxes': boxes, 'labels': labels, 'image_id': torch.tensor([self.img_idx[index]])}

        # transform
        if self.transforms:
            while True:
                sample = self.transforms(**{
                    'image': image,
                    'bboxes': target['boxes'],
                    'labels': labels
                })
                if len(sample['bboxes']) > 0:
                    image = sample['image']
                    target['boxes'] = torch.stack(tuple(map(torch.tensor, zip(*sample['bboxes'])))).permute(1, 0)
                    target['boxes'][:,[0,1,2,3]] = target['boxes'][:,[1,0,3,2]]  #yxyx: be warning
                    target['labels'] = torch.tensor(sample['labels'])
                    break

        return image, target, image_id

    def load_image_boxes_labels(self, index):

        image_id = self.coco.getImgIds(imgIds = self.img_idx[index])

        image_info = self.coco.loadImgs(image_id)[0]

        image = cv2.imread(os.path.join(self.data_dir, image_info['file_name']))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0

        ann_ids = self.coco.getAnnIds(imgIds=image_info['id'])
        anns = self.coco.loadAnns(ann_ids)

        boxes = np.array([x['bbox'] for x in anns])

        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]

        labels = np.array([x['category_id'] + 1 for x in anns]) 
        labels = torch.as_tensor(labels, dtype=torch.int64)

        return image, boxes, labels, image_id


    def load_cutmix(self, index, img_size = 1024):

        w, h = img_size, img_size
        s = img_size // 2

        image_id = self.coco.getImgIds(imgIds=self.img_idx[index])
        
        xc, yc = [int(random.uniform(img_size * 0.25, img_size * 0.75)) for _ in range(2)] 
        indexes = [index] + [random.randint(0, len(self.img_idx) - 1) for _ in range(3)]

        result_image = np.full((img_size, img_size, 3), 1, dtype = np.float32)
        result_boxes = []
        result_labels = []

        for i, index in enumerate(indexes):
            image, boxes, labels, _ = self.load_image_boxes_labels(index)

            if i == 0:
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, max(xc, w), min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

            result_image[y1a:y2a, x1a:x2a] = image[y1b:y2b, x1b:x2b]
            padw = x1a - x1b
            padh = y1a - y1b

            boxes[:, 0] += padw
            boxes[:, 1] += padh
            boxes[:, 2] += padw
            boxes[:, 3] += padh

            result_boxes.append(boxes)
            result_labels.append(labels)

        result_boxes = np.concatenate(result_boxes, 0)
        result_labels = np.concatenate(result_labels, 0)

        np.clip(result_boxes[:, 0:], 0, 2 * s, out = result_boxes[:, 0:])
        result_boxes = result_boxes.astype(np.float32)
        new_indexes = np.where((result_boxes[:,2]-result_boxes[:,0])*(result_boxes[:,3]-result_boxes[:,1]) > 0)

        result_boxes = result_boxes[new_indexes]
        result_labels = result_labels[new_indexes]

        return result_image, result_boxes, result_labels, image_id


    def load_mixup(self, index):

        image_id = self.coco.getImgIds(imgIds=self.img_idx[index])
        indexes = [index, random.randint(0, len(self.img_idx) - 1)]

        image1, boxes1, labels1, _ = self.load_image_boxes_labels(indexes[0])
        image2, boxes2, labels2, _ = self.load_image_boxes_labels(indexes[1])

        result_image = (image1 + image2) / 2
        result_labels = [labels1, labels2]
        result_boxes = [boxes1, boxes2]

        result_labels = np.concatenate(result_labels, 0)
        result_boxes = np.concatenate(result_boxes, 0)

        return result_image, result_boxes, result_labels, image_id
        

    def _area_func(self, x):

        linspace = np.linspace(0, 1024*1024, 11)
        if x < linspace[1]:
            return 0
        elif x < linspace[2]:
            return 1
        elif x < linspace[3]:
            return 2
        elif x < linspace[4]:
            return 3
        elif x < linspace[5]:
            return 4
        elif x < linspace[6]:
            return 5
        elif x < linspace[7]:
            return 6
        elif x < linspace[8]:
            return 7
        elif x < linspace[9]:
            return 8
        else:
            return 9
        

class TestDataset(Dataset):

    def __init__(self, annotation, data_dir, transforms=None):
        super().__init__()
        self.data_dir = data_dir
        # coco annotation 불러오기 (coco API)
        self.coco = COCO(annotation)
        self.predictions = {
            "images": self.coco.dataset["images"].copy(),
            "categories": self.coco.dataset["categories"].copy(),
            "annotations": None
        }
        self.transforms = transforms


    def __getitem__(self, index: int):
        image_id = self.coco.getImgIds(imgIds=index)

        image_info = self.coco.loadImgs(image_id)[0]
        
        image = cv2.imread(os.path.join(self.data_dir, image_info['file_name']))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0
        
        if self.transforms:
            sample = self.transforms(image=image)

        return sample['image'], image_id
    

    def __len__(self) -> int:

        return len(self.coco.getImgIds())