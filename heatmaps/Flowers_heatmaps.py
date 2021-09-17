
import sys
sys.path.append('..')

import os
import torch
import scipy.io
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter
from Visualization.VIT_LRP import VisionTransformer, _conv_filter
from Visualization.generator import image2heatmap


###### Function: Generate heatmaps of Oxford Flowers ######
###### All heatmaps are stored in the folder named 'heatmaps' ######


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, default='data/Oxford_flowers',help='The path of dataset.')

    parser.add_argument('--model_path', type=str, default='vit_pretrained.pth',help='The path of pretrained model.')
    args = parser.parse_args()
    return args


args = parse_args()

root_path = args.root_path
model_path = args.model_path

images_root_path = os.path.join(root_path, 'jpg')
split_path = os.path.join(root_path, 'setid.mat')
heatmaps_path = os.path.join(root_path, 'heatmaps')

images_path = os.listdir(images_root_path)
split = scipy.io.loadmat(split_path)

image_index = split['trnid'][0]-1


def make_dirs():
    if not os.path.exists(heatmaps_path):
        os.makedirs(heatmaps_path)


def vit_base_patch16_384():
    model = VisionTransformer(img_size=384, patch_size=16, embed_dim=768,
                              depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True, num_classes=102)
    # The rootpath of pretrained ViT
    checkpoint = torch.load(
        model_path, map_location='cpu')
    checkpoint = _conv_filter(checkpoint)
    model.load_state_dict(checkpoint, strict=True)
    return model


net = vit_base_patch16_384().to(device)


def generate_heatmaps(image_index):
    for index in tqdm(image_index):
        pic_path = os.path.join(images_root_path, images_path[index])
        save_path = os.path.join(heatmaps_path, images_path[index])

        img = image2heatmap(net, pic_path)
        img = Image.fromarray(img.astype('uint8')).convert('RGB')

        img = img.filter(ImageFilter.GaussianBlur(2))
        fig = plt.gcf()

        plt.imshow(img)
        plt.axis('off')
        fig.set_size_inches(438/100., 438/100.)

        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top=1, bottom=0, left=0,
                            right=1, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.savefig(save_path)
        plt.clf()


make_dirs()
generate_heatmaps(image_index)
