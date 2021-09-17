
import sys
sys.path.append('..')

import os
import torch
import scipy.io
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image, ImageFilter
from Visualization.generator import image2heatmap
from Visualization.VIT_LRP import VisionTransformer, _conv_filter


###### Function: Generate heatmaps of Stanford Cars ######
###### All heatmaps are stored in the folder named 'heatmaps' ######


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str,
                        default='data/Stanford_Cars', help='The path of dataset.')

    parser.add_argument('--model_path', type=str,
                        default='vit_pretrained.pth', help='The path of pretrained model.')
    args = parser.parse_args()
    return args


args = parse_args()


root_path = args.root_path
model_path = args.model_path

mat_path = os.path.join(root_path, 'cars_annos.mat')
pic_root_path = os.path.join(root_path, 'car_ims')
heatmaps_path = os.path.join(root_path, 'heatmaps')

loaded_mat = scipy.io.loadmat(mat_path)['annotations'][0]
split = []
for item in loaded_mat:
    if bool(item[-1][0]) != True:
        path = str(item[0][0])
        path = path[8:]
        split.append(path)


def make_dirs():
    if not os.path.exists(heatmaps_path):
        os.makedirs(heatmaps_path)


def vit_base_patch16_384():
    model = VisionTransformer(img_size=384, patch_size=16, embed_dim=768,
                              depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True, num_classes=196)
    # The rootpath of pretrained ViT
    checkpoint = torch.load(
        model_path, map_location='cpu')
    checkpoint = _conv_filter(checkpoint)
    model.load_state_dict(checkpoint, strict=True)
    return model


net = vit_base_patch16_384().to(device)


def generate_heatmaps(split):
    for line in tqdm(split):
        pic_path = os.path.join(pic_root_path, line)
        save_path = os.path.join(heatmaps_path, line)

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
generate_heatmaps(split)
