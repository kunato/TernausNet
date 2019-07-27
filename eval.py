from main import HumanSegmentationDataset, variable
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
        model = LinkNet34(num_classes=1,num_channels=3)
    state = torch.load(f'models/{model_name}/best-model_{fold}.pt')

    model.load_state_dict(state['model'])
    model.eval()
    return model

def visualize(full_image, mask, show_light=True, show_base=True):
    mask = np.tile(mask, (3,1,1))
    mask = np.moveaxis(mask, 0, -1)
    full_image = np.moveaxis(full_image, 0, -1) 
    if full_image is not None and show_light:
        print(full_image.shape, mask.shape)
        light_heat = cv2.addWeighted(full_image, 0.6, mask, 0.4, 0)
        print(light_heat.shape)
        cv2.imshow('light heat', light_heat)
    
    if full_image is not None and show_base:
        cv2.imshow('image', full_image)
        cv2.imshow('mask', mask)
    if show_light or show_base:
        cv2.waitKey()

def eval():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--model', default='linknet', help='model name')
    arg('--fold', type=int, help='fold', default=0)
    args = parser.parse_args()


    model = get_model(args.model, args.fold)
    print('model loaded')
    loader = DataLoader(
        dataset=HumanSegmentationDataset(f'val_{args.fold}.txt'),
        shuffle=False,
        batch_size=1,
        num_workers=1,
        pin_memory=True
    )
    print('data loaded')


    with torch.no_grad():
        for batch_num, (inputs, gt_mask) in enumerate(tqdm(loader)):
            print(inputs)
            input_img = (inputs[0].data.cpu().numpy() * 255).astype(np.uint8)
            print(input_img[0, :,:])
            # cv2.imwrite(f'{batch_num}_test.png', input_img[0, :,:])
            inputs = variable(inputs, volatile=False)
            outputs = model(inputs)
            out_mask = (outputs.data.cpu().numpy() * 255).astype(np.uint8)
            for i, result_mask in enumerate(out_mask):
                input = (inputs[i].data.cpu().numpy() * 255).astype(np.uint8)
                cv2.imwrite(f'result/{batch_num}_{i}.png',  result_mask[0, :, :])
                # visualize(input, result_mask)
    print(args.model)



if __name__ == "__main__":
    eval()