import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transform():
    ops = [
        A.Resize(1024, 1024),
         A.OneOf([
                A.HueSaturationValue(hue_shift_limit=0.5, 
                                     sat_shift_limit= 0.5,
                                     val_shift_limit=0.5,
                                     p=0.9),
                A.RandomBrightnessContrast(brightness_limit=0.5, 
                                           contrast_limit=0.2, 
                                           p=0.9),
            ],p=0.9),
        A.ToGray(p=0.05),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Cutout(num_holes=8, max_h_size=64, max_w_size=64, fill_value=0, p=0.5),
        ToTensorV2(p = 1.0)
    ]
    transforms = A.Compose(ops, bbox_params = {'format': 'pascal_voc', 'label_fields' : ['labels']})

    return transforms


def get_valid_transform():

    return A.Compose([
        A.Resize(1024, 1024, p = 1),
        ToTensorV2(p=1.0)
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})


def get_test_transform():

    return A.Compose([
        A.Resize(1024, 1024, p = 1),
        ToTensorV2(p=1.0)
    ])


def collate_fn(batch):
    
    return tuple(zip(*batch))