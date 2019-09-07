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
from model.unet import UNet11
from model.unet import Loss as UNet11Loss
from model.linknet import LinkNet34
from model.linknet import BCEDiceLoss as LinkNet34Loss
from model.unet_fpa import Res34Unetv5
# from model.deeplab import Res_Deeplab
from model.psp import PSPNet50, PSPNet34
import random
import tqdm
import cv2
import json
from datetime import datetime

cuda_is_available = torch.cuda.is_available()
device = torch.device('cuda:1') if cuda_is_available else torch.device('cpu')

RESOLUTION = {
    "deeplabv3": (576, 384),
    "pspnet50": (576, 384),
    "pspnet34": (1024, 768),
    "unet34fpa": (576, 384),
    "linknet34": (1024, 768),
    "unet11": (1024, 768)
}

class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor

# save_predictions(root, p_i, inputs, targets, outputs)
def save(root, i, imgs, targets, outputs):
    unorm = UnNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    imgs = unorm(imgs)
    imgs = imgs.permute(0,2,3,1)
    output = outputs.cpu().detach().numpy()[0, 0, :, :]
    output[output > 0.5] = 1.0
    output[output <= 0.5] = 0.0
    output = (output * 255).astype(np.uint8)
    img = (imgs.cpu().detach().numpy() * 255).astype(np.uint8)[0, :, :, :]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(root / 'out' / f'{i}_img.png'), img)
    cv2.imwrite(str(root / 'out' / f'{i}_out.png'), output)
    img2 = np.stack((output,)*3, axis=-1).astype(np.uint8)
    dst = cv2.addWeighted(img, 0.5, img2, 0.5, 0)
    cv2.imwrite(str(root / f'out/{i}_result.png'), dst)
    

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


def variable(x):

    def cuda(x):
        return x.to(device)

    if isinstance(x, (list, tuple)):
        return [variable(y) for y in x]
    return cuda(Variable(x))


img_transform = Compose([
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def load_image(path, model):
    img = cv2.imread(path)
    try:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if img is None:
            print(img, path)
        
        img = cv2.resize(img, RESOLUTION[model])
        return img.astype(np.uint8)
    except:
        print(f'have some problem {path}')
        return img


def load_mask(path, model):
    img = cv2.imread(path, -1)  # it's RGBA image
    img = cv2.resize(img, RESOLUTION[model])
    a = img[:, :]
    a[(a > 5)] = 255
    a[(a <= 5)] = 0
    a = (a / 255).astype(np.float32)
    return a


class PeopleSegmentationDataset(Dataset):

    def __init__(self, file_path, args):
        with open(file_path, 'r') as f:
            images = f.readlines()
            self.images = images
            self.model = args.model

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        mask = img.replace('/image/', '/alpha/')
        img = load_image(img.strip(), self.model)
        mask = load_mask(mask.strip(), self.model)
        return img_transform(img), torch.from_numpy(np.expand_dims(mask, 0))


def cyclic_lr(epoch,
              init_lr=1e-4,
              num_epochs_per_cycle=5,
              cycle_epochs_decay=2,
              lr_decay_factor=0.5):
    epoch_in_cycle = epoch % num_epochs_per_cycle
    lr = init_lr * (lr_decay_factor**(epoch_in_cycle // cycle_epochs_decay))
    return lr


def validation(args, model: nn.Module, criterion, valid_loader, epoch, save_predictions=None) -> Dict[str, float]:
    model.eval()
    with torch.no_grad():
        losses = []
        dice = []
        i = 0
        for inputs, targets in valid_loader:
            inputs = variable(inputs)
            targets = variable(targets)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            losses.append(loss.item())
            
            dice += [get_dice(targets, (outputs > 0.5).float()).item()]
            
            if save_predictions:
                root = Path(args.root)
                save_predictions(root, f'test_{epoch}_{i}', inputs, targets, outputs)
            i+= 1

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

    save_model = lambda ep: torch.save(
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
                        save_predictions(root, f'train_{epoch}_{p_i}', inputs, targets, outputs)
            write_event(log, step, loss=mean_loss)
            tq.close()
            save_model(epoch + 1)
            valid_metrics = validation(args, model, criterion, valid_loader, epoch, save_predictions=save_predictions)
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
            save_model(epoch)
            print('done.')
            return


def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--model', default='linknet34', help='model')
    arg('--batch-size', type=int, default=4)
    arg('--n-epochs', type=int, default=200)
    arg('--workers', type=int, default=5)
    arg('--epoch-size', type=int)
    arg('--fold', type=int, help='fold', default=0)

    args = parser.parse_args()
    model = args.model
    args.root = f'models/{model}'
    if model == 'unet11':
        model = UNet11().to(device) if cuda_is_available else UNet11()
        loss = UNet11Loss()
        print('arch as Unet11')
    elif model == 'linknet34':
        model = LinkNet34(
            num_classes=1,
            num_channels=3).to(device) if cuda_is_available else LinkNet34(
                num_classes=1, num_channels=3)
        loss = LinkNet34Loss().to(device) if cuda_is_available else LinkNet34Loss()
        # loss = UNet11Loss()
        print('arch as Linknet34')
    # elif model == 'deeplabv3':
    #     model = Res_Deeplab(num_classes=1).to(device)
    #     loss = UNet11Loss()
    elif model == 'pspnet50':
        model = PSPNet50(num_classes=1).to(device)
        loss = UNet11Loss()
        print('arch as pspnet50')
    elif model == 'pspnet34':
        model = PSPNet34(num_classes=1).to(device)
        loss = UNet11Loss()
        print('arch as pspnet34')
    elif model == 'unet34fpa':
        model = Res34Unetv5().to(device)
        loss = UNet11Loss()
        print('arch as unet34fpa')

    def make_loader(file_path, args, shuffle=False):
        return DataLoader(dataset=PeopleSegmentationDataset(file_path, args),
                          shuffle=shuffle,
                          num_workers=args.workers,
                          batch_size=args.batch_size,
                          pin_memory=True, drop_last=True)

    train_loader = make_loader(f'/home/kunato/dataset/Supervisely_Person_Dataset/train.txt', args, shuffle=True)
    valid_loader = make_loader(f'/home/kunato/dataset/Supervisely_Person_Dataset/test.txt', args)

    optimz = lambda lr: Adam(model.parameters(), lr=lr)
    # optimz = lambda lr: RMSprop(model.parameters(), lr=lr)

    train(init_optimizer=optimz,
          args=args,
          model=model,
          criterion=loss,
          train_loader=train_loader,
          valid_loader=valid_loader,
          validation=validation,
          save_predictions=save,
          fold=args.fold)


if __name__ == "__main__":
    main()
