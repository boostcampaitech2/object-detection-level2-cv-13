import torch


class Averager:
    def __init__(self):
        self.current_total = 0.0
        self.iterations = 0.0

    def send(self, value):
        self.current_total += value
        self.iterations += 1

    @property
    def value(self):
        if self.iterations == 0:
            return 0
        else:
            return 1.0 * self.current_total / self.iterations

    def reset(self):
        self.current_total = 0.0
        self.iterations = 0.0


def collate_data(images, targets, device):

    images = torch.stack(images)
    images = images.to(device).float()
    
    boxes = [target['boxes'].to(device).float() for target in targets]
    labels = [target['labels'].to(device).float() for target in targets]
    targets = {"bbox": boxes, "cls": labels}

    return images, targets


def get_loss(model, images, targets, device):
    
    images, targets = collate_data(images, targets, device)
    loss, _, _ = model(images, targets).values()

    return loss