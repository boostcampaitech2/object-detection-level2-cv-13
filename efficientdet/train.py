from map_boxes import mean_average_precision_for_boxes
from models import efficientdet
from effdet import unwrap_bench, DetBenchTrain, DetBenchPredict
from utils import Averager, collate_data, get_loss, arg_parse
from tqdm import tqdm
import torch
import os
from datasets import TrainDataset
from transforms import get_train_transform, collate_fn, get_valid_transform
from torch.utils.data import DataLoader


def train_fn(model, train_dataloader, optimizer, device, clip = None):

    model = unwrap_bench(model)
    model = DetBenchTrain(model)
    model.to(device)
    model.train()
    loss_hist = Averager()

    for images, targets, image_id in tqdm(train_dataloader):

        loss = get_loss(model, images, targets, device)
        loss_value = loss.detach().item()

        loss_hist.send(loss_value)

        # backward
        optimizer.zero_grad()
        loss.backward()

        if clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

    return loss_hist.value


def eval_fn(model, val_dataloader, device, threshold = 0.05):
    model = unwrap_bench(model)
    model = DetBenchPredict(model)
    model.to(device)
    model.eval()

    gt_list = []
    pred_list = []

    for images, targets, image_id in tqdm(val_dataloader):

        images, targets = collate_data(images, targets, device)

        output = model(images)
        for i in range(len(output)):

            boxes = output[i][:, :4].detach().cpu().numpy()
            scores = output[i][:, 4].detach().cpu().numpy()
            labels = output[i][:, -1].detach().cpu().numpy()

            for label, score, box, in zip(labels, scores, boxes):
                if score > threshold:
                    pred_list.append([str(image_id[i][0]), str(label-1), score, box[0], box[2], box[1], box[3]])

        for i in range(len(output)):

            for label, box in zip(targets['cls'][i].detach().cpu().numpy(), targets['bbox'][i].detach().cpu().numpy()):
                gt_list.append([str(image_id[i][0]), str(label-1), box[1], box[3], box[0], box[2]])

    mean_ap, average_precisions = mean_average_precision_for_boxes(gt_list, pred_list, iou_threshold=0.5, verbose = False)
    print(average_precisions)
    return mean_ap


def main():
    cfgs = arg_parse()
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    annotation = cfgs['train_annotation']
    data_dir = cfgs['data_dir']
    checkpoint = cfgs['checkpoint']
    train_ds_cfgs = cfgs['train_dataset']
    val_ds_cfgs = cfgs['val_dataset']
    train_dl_cfgs = cfgs['train_dataloader']
    val_dl_cfgs = cfgs['val_dataloader']
    opt_cfgs = cfgs['optimizer']
    num_epochs = cfgs['epochs']

    train_transforms = get_train_transform()
    val_transforms = get_valid_transform()

    train_dataset = TrainDataset(annotation, data_dir,
                                 train_ds_cfgs['mode'],
                                 train_ds_cfgs['cutmix'],
                                 train_ds_cfgs['mixup'],
                                 train_ds_cfgs['fold'],
                                 train_ds_cfgs['k'],
                                 train_ds_cfgs['random_state'], 
                                 transforms = train_transforms)

    val_dataset = TrainDataset(annotation, data_dir,
                               val_ds_cfgs['mode'],
                               val_ds_cfgs['cutmix'],
                               val_ds_cfgs['mixup'],
                               val_ds_cfgs['fold'],
                               val_ds_cfgs['k'],
                               val_ds_cfgs['random_state'], 
                               transforms = val_transforms)

    train_dataloader = DataLoader(
            train_dataset,
            batch_size = train_dl_cfgs['batch_size'],
            shuffle = train_dl_cfgs['shuffle'],
            num_workers = train_dl_cfgs['num_workers'],
            collate_fn = collate_fn)

    val_dataloader = DataLoader(
            val_dataset,
            batch_size = val_dl_cfgs['batch_size'],
            shuffle = val_dl_cfgs['shuffle'],
            num_workers = val_dl_cfgs['num_workers'],
            collate_fn = collate_fn)
    
    model = efficientdet()
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]

    optimizer = torch.optim.AdamW(params, lr = opt_cfgs['lr'])
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           mode = 'max',
                                                           factor = 0.5,
                                                           patience = 3)

    best_val_map = 0
    for epoch in range(num_epochs):
        
        train_loss = train_fn(model, train_dataloader, optimizer, device)
        print(f"Epoch #{epoch} train_loss: {train_loss}")
        
        val_map = eval_fn(model, val_dataloader, device, 0.05)
        print(f"eval_map: {val_map}")
        
        scheduler.step(val_map)
        if val_map > best_val_map:
            save_path = checkpoint.format(epoch, val_map)
            save_dir = os.path.dirname(save_path)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            torch.save(model.state_dict(), save_path)
            best_val_map = val_map


if __name__ == "__main__":
    main()