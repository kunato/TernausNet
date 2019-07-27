import glob
import argparse
from typing import Dict
import numpy as np
import torch
from pathlib import Path
from torch import nn
from torch.optim import Adam, RMSprop
from torch.autograd import Variable
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor, Normalize, Compose
from unet_models import UNet11
from unet_models import Loss as UNet11Loss
from linknet_models import LinkNet34
from linknet_models import BCEDiceLoss as LinkNet34Loss
import random
import tqdm
import cv2
import json
from datetime import datetime

cuda_is_available = torch.cuda.is_available()


def get_dice(y_true, y_pred):
    epsilon = 1e-15
    intersection = (y_pred * y_true).sum(dim=-2).sum(dim=-1)
    union = y_true.sum(dim=-2).sum(dim=-1) + y_pred.sum(dim=-2).sum(
        dim=-1) + epsilon

    return 2 * (intersection / union).mean()


def write_event(log, step: int, **data):
    data['step'] = step
    data['dt'] = datetime.now().isoformat()
    log.write(json.dumps(data, sort_keys=True))
    log.write('\n')
    log.flush()


def variable(x, volatile=None):

    def cuda(x):
        if volatile is not None:
            return x.cuda() if volatile else x
        return x.cuda() if cuda_is_available else x

    if isinstance(x, (list, tuple)):
        return [variable(y) for y in x]
    return cuda(Variable(x))


img_transform = Compose([
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def load_image(path):
    img = cv2.imread(path)
    try:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if img is None:
            print(img, path)
        img = cv2.resize(img, (768, 576))
        return img.astype(np.uint8)
    except:
        print(f'have some problem {path}')
        return img


def load_mask(path):
    img = cv2.imread(path, -1)  # it's RGBA image
    img = cv2.resize(img, (768, 576))
    a = img[:, :, 3]
    a[(a > 5)] = 255
    a[(a <= 5)] = 0
    a = (a / 255).astype(np.float32)
    return a


class HumanSegmentationDataset(Dataset):

    def __init__(self, file_path):
        with open(file_path, 'r') as f:
            fgs = f.readlines()
            self.fgs = fgs

    def __len__(self):
        return len(self.fgs)

    def __getitem__(self, idx):
        fg = self.fgs[idx]
        bg = fg.replace('/matting/', '/clip_img/').replace(
            '/matting_0', '/clip_0').replace('.png', '.jpg')
        img = load_image(bg.strip())
        mask = load_mask(fg.strip())
        return img_transform(img), torch.from_numpy(np.expand_dims(mask, 0))


def cyclic_lr(epoch,
              init_lr=1e-4,
              num_epochs_per_cycle=5,
              cycle_epochs_decay=2,
              lr_decay_factor=0.5):
    epoch_in_cycle = epoch % num_epochs_per_cycle
    lr = init_lr * (lr_decay_factor**(epoch_in_cycle // cycle_epochs_decay))
    return lr


def validation(model: nn.Module, criterion, valid_loader) -> Dict[str, float]:
    model.eval()
    with torch.no_grad():
        losses = []
        dice = []
        for inputs, targets in valid_loader:
            inputs = variable(inputs)
            targets = variable(targets)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            losses.append(loss.item())
            outputs_sigmoid = torch.sigmoid(outputs)
            dice += [get_dice(targets, (outputs_sigmoid > 0.5).float()).item()]

        valid_loss = np.mean(losses)  # type: float

        valid_dice = np.mean(dice)

        print('Valid loss: {:.5f}, dice: {:.5f}'.format(valid_loss, valid_dice))
        metrics = {'valid_loss': valid_loss, 'dice_loss': valid_dice}
        return metrics


def train(args,
          model: nn.Module,
          criterion,
          *,
          train_loader,
          valid_loader,
          validation,
          init_optimizer,
          fold=None,
          save_predictions=None,
          n_epochs=None):
    n_epochs = n_epochs or args.n_epochs

    root = Path(args.root)
    model_path = root / 'model_{fold}.pt'.format(fold=fold)
    best_model_path = root / 'best-model_{fold}.pt'.format(fold=fold)
    if model_path.exists():
        state = torch.load(str(model_path))
        epoch = state['epoch']
        step = state['step']
        best_valid_loss = state['best_valid_loss']
        model.load_state_dict(state['model'])
        print('Restored model, epoch {}, step {:,}'.format(epoch, step))
    else:
        epoch = 1
        step = 0
        best_valid_loss = float('inf')

    save = lambda ep: torch.save(
        {
            'model': model.state_dict(),
            'epoch': ep,
            'step': step,
            'best_valid_loss': best_valid_loss
        }, str(model_path))

    report_each = 10
    save_prediction_each = report_each * 20
    log = root.joinpath('train_{fold}.log'.format(fold=fold)).open(
        'at', encoding='utf8')
    valid_losses = []

    for epoch in range(epoch, n_epochs + 1):
        lr = cyclic_lr(epoch)

        optimizer = init_optimizer(lr)

        model.train()

        random.seed()
        tq = tqdm.tqdm(
            total=(args.epoch_size or len(train_loader) * args.batch_size))
        tq.set_description('Epoch {}, lr {}'.format(epoch, lr))

        losses = []
        try:
            mean_loss = 0
            for i, (inputs, targets) in enumerate(train_loader):
                inputs, targets = variable(inputs), variable(targets)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                optimizer.zero_grad()
                batch_size = inputs.size(0)
                step += 1
                tq.update(batch_size)
                losses.append(loss.item())
                mean_loss = np.mean(losses[-report_each:])
                tq.set_postfix(loss='{:.5f}'.format(mean_loss))

                (batch_size * loss).backward()
                optimizer.step()

                if i and i % report_each == 0:
                    write_event(log, step, loss=mean_loss)
                    if save_predictions and i % save_prediction_each == 0:
                        p_i = (i // save_prediction_each) % 5
                        save_predictions(root, p_i, inputs, targets, outputs)
            write_event(log, step, loss=mean_loss)
            tq.close()
            save(epoch + 1)
            valid_metrics = validation(model, criterion, valid_loader)
            write_event(log, step, **valid_metrics)
            valid_loss = valid_metrics['valid_loss']
            valid_losses.append(valid_loss)
            if valid_loss < best_valid_loss:
                import shutil
                print(f'Save epoch {epoch}, valid loss : {valid_loss}')
                best_valid_loss = valid_loss
                shutil.copy(str(model_path), str(best_model_path))
        except KeyboardInterrupt:
            tq.close()
            print('Ctrl+C, saving snapshot')
            save(epoch)
            print('done.')
            return


def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--model', default='linknet', help='model')
    arg('--batch-size', type=int, default=4)
    arg('--n-epochs', type=int, default=100)
    arg('--lr', type=float, default=0.0001)
    arg('--workers', type=int, default=5)
    arg('--epoch-size', type=int)
    arg('--fold', type=int, help='fold', default=0)

    args = parser.parse_args()
    model = args.model
    args.root = f'models/{model}'
    if model == 'unet_11':
        model = UNet11().cuda() if cuda_is_available else UNet11()
        loss = UNet11Loss()
        print('arch as Unet11')
    elif model == 'linknet':
        model = LinkNet34(
            num_classes=1,
            num_channels=3).cuda() if cuda_is_available else LinkNet34(
                num_classes=1, num_channels=3)
        loss = LinkNet34Loss().cuda() if cuda_is_available else LinkNet34Loss()
        print('arch as Linknet34')

    def make_loader(file_path, shuffle=False):
        return DataLoader(dataset=HumanSegmentationDataset(file_path),
                          shuffle=shuffle,
                          num_workers=args.workers,
                          batch_size=args.batch_size,
                          pin_memory=True)

    train_loader = make_loader(f'train_{args.fold}.txt', shuffle=True)
    valid_loader = make_loader(f'test_{args.fold}.txt')

    optimz = lambda lr: Adam(model.parameters(), lr=lr)
    # optimz = lambda lr: RMSprop(model.parameters(), lr=lr)

    train(init_optimizer=optimz,
          args=args,
          model=model,
          criterion=loss,
          train_loader=train_loader,
          valid_loader=valid_loader,
          validation=validation,
          fold=args.fold)


if __name__ == "__main__":
    main()
