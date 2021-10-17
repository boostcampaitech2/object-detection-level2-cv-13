from datasets import TestDataset
from pycocotools.coco import COCO
from tqdm import tqdm
import pandas as pd
from torch.utils.data import DataLoader
import torch
from models import efficientdet
from effdet import unwrap_bench, DetBenchPredict
from transforms import collate_fn, get_test_transform
from utils import arg_parse


def inference_fn(test_data_loader, model, device):
    
    model = unwrap_bench(model)
    model = DetBenchPredict(model)
    model.to(device)
    model.eval()

    outputs = []
    for images, image_ids in tqdm(test_data_loader):

        images = torch.stack(images) # bs, ch, w, h 
        images = images.to(device).float()
        output = model(images)
        for out in output:
            outputs.append({'boxes': out.detach().cpu().numpy()[:,:4], 
                            'scores': out.detach().cpu().numpy()[:,4], 
                            'labels': out.detach().cpu().numpy()[:,-1]})
    return outputs


def make_submission(outputs, annotation, submission_path, score_threshold):
    coco = COCO(annotation)
    prediction_strings = []
    file_names = []

    for i, output in enumerate(outputs):
        prediction_string = ''
        image_info = coco.loadImgs(coco.getImgIds(imgIds=i))[0]
        for box, score, label in zip(output['boxes'], output['scores'], output['labels']):
            if score > score_threshold: 
                # label[1~10] -> label[0~9]
                prediction_string += str(label-1) + ' ' + str(score) + ' ' + str(box[0]) + ' ' + str(
                    box[1]) + ' ' + str(box[2]) + ' ' + str(box[3]) + ' '
        prediction_strings.append(prediction_string)
        file_names.append(image_info['file_name'])
    submission = pd.DataFrame()
    submission['PredictionString'] = prediction_strings
    submission['image_id'] = file_names
    submission.to_csv(submission_path, index=None)


def main():

    cfgs = arg_parse()

    annotation = cfgs['test_annotation']
    data_dir = cfgs['data_dir']
    val_dl_cfgs = cfgs['val_dataloader']
    best_model_dir = cfgs['best_model']
    submission_dir = cfgs['submission']

    test_transforms = get_test_transform()
    test_dataset = TestDataset(annotation, data_dir, test_transforms)
    score_threshold = 0.05

    val_data_loader = DataLoader(
        test_dataset,
        batch_size = val_dl_cfgs['batch_size'],
        shuffle = val_dl_cfgs['shuffle'],
        num_workers = val_dl_cfgs['num_workers'],
        collate_fn=collate_fn
    )
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = efficientdet()
    model.load_state_dict(torch.load(best_model_dir))
    model = DetBenchPredict(model)
    model.to(device)

    outputs = inference_fn(val_data_loader, model, device)
    
    prediction_strings = []
    file_names = []
    coco = COCO(annotation)
    
    for i, output in enumerate(outputs):
        prediction_string = ''
        image_info = coco.loadImgs(coco.getImgIds(imgIds=i))[0]
        for box, score, label in zip(output['boxes'], output['scores'], output['labels']):
            if score > score_threshold:
                prediction_string += str(int(label) - 1) + ' ' + str(score) + ' ' + str(box[0]) + ' ' + str(
                    box[1]) + ' ' + str(box[2]) + ' ' + str(box[3]) + ' '
        prediction_strings.append(prediction_string)
        file_names.append(image_info['file_name'])
        
    submission = pd.DataFrame()
    submission['PredictionString'] = prediction_strings
    submission['image_id'] = file_names
    submission.to_csv(submission_dir, index=None)


if __name__ == "__main__":
    main()