from main import HumanSegmentationDataset, variable, load_image
import torch
import numpy as np
import cv2
from unet_models import UNet11
from linknet_models import LinkNet34
import argparse
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

cuda_is_available = False


def get_model(model_name, fold):
    if model_name == 'unet_11':
        model = UNet11()
    if model_name == 'linknet':
        model = LinkNet34(num_classes=1, num_channels=3)
    state = torch.load(f'models/{model_name}/best-model_{fold}.pt')

    model.load_state_dict(state['model'])
    model.eval()
    return model


def visualize(full_image, mask, show_light=True, show_base=True):
    mask = np.tile(mask, (3, 1, 1))
    mask = np.moveaxis(mask, 0, -1)
    if full_image is not None and show_light:
        print(full_image.shape, mask.shape)
        light_heat = cv2.addWeighted(full_image, 0.6, mask, 0.4, 0)
        print(light_heat.shape)
        cv2.imshow('light heat', light_heat)

    if full_image is not None and show_base:
        cv2.imshow('image', full_image)
        cv2.imshow('mask', mask)
    cv2.waitKey(0)

def load_raw_img(file_path, idx):
    with open(file_path, 'r') as f:
        fgs = f.readlines()
        bg = fgs[idx].strip().replace('/matting/', '/clip_img/').replace(
            '/matting_0', '/clip_0').replace('.png', '.jpg')
        img = load_image(bg)
        return img



def eval():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--model', default='linknet', help='model name')
    arg('--fold', type=int, help='fold', default=0)
    args = parser.parse_args()

    model = get_model(args.model, args.fold)
    print('model loaded')

    input_file_list = f'val_{args.fold}.txt'


    loader = DataLoader(
        dataset=HumanSegmentationDataset(input_file_list),
        shuffle=False,
        batch_size=1,
        num_workers=1,
        pin_memory=False)
    print('data loaded')

    with torch.no_grad():
        for batch_num, (inputs, gt_mask) in enumerate(tqdm(loader)):
            input_img = (inputs[0].data.cpu().numpy() * 255).astype(np.uint8)
            # cv2.imwrite(f'{batch_num}_test.png', input_img[0, :,:])
            inputs = variable(inputs, volatile=False)
            outputs = model(inputs)
            if args.model == 'linknet':
                outputs = torch.sigmoid(outputs)
            out_mask = (outputs.data.cpu().numpy() * 255).astype(np.uint8)
            for i, result_mask in enumerate(out_mask):
                raw_img = load_raw_img(input_file_list, batch_num * 1 + i)
                cv2.imwrite(f'result/{batch_num}_{i}.png', result_mask[0, :, :])
                visualize(raw_img, result_mask)
    print(args.model)


if __name__ == "__main__":
    eval()