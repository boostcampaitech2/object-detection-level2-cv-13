import torch
from effdet import get_efficientdet_config, EfficientDet
from effdet.efficientdet import HeadNet


def efficientdet(checkpoint_path=None):
    
    config = get_efficientdet_config('tf_efficientdet_d4')
    config.num_classes = 10
    config.image_size = (1024, 1024)
    
    config.soft_nms = False
    config.max_det_per_image = 100
    
    net = EfficientDet(config, pretrained_backbone=True)
    net.class_net = HeadNet(config, num_outputs=config.num_classes)
    
    if checkpoint_path:
        net.load_state_dict(torch.load(checkpoint_path))
    
    return net