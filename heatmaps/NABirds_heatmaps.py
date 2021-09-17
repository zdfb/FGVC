
import sys
sys.path.append('..')

import os
import torch
import argparse
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter
from Visualization.generator import image2heatmap
from Visualization.VIT_LRP import VisionTransformer, _conv_filter


###### Function: Generate heatmaps of NABirds ######
###### All heatmaps are stored in the folder named 'heatmaps' ######


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, default='data/NABirds', help='The path of dataset.')

    parser.add_argument('--model_path', type=str, default='vit_pretrained.pth',help='The path of pretrained model.')
    args = parser.parse_args()
    return args


args = parse_args()

root_path =  args.root_path 
model_path = args.model_path

pic_root_path = os.path.join(root_path, 'images')
heatmaps_path = os.path.join(root_path, 'heatmaps')


def make_dirs():
    if not os.path.exists(heatmaps_path):
        os.makedirs(heatmaps_path)

    pic_dirs = os.listdir(pic_root_path)
    for pic_dir in pic_dirs:
        dir_path = os.path.join(heatmaps_path, pic_dir)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)


def vit_base_patch16_384():
    model = VisionTransformer(img_size=384, patch_size=16, embed_dim=768,
                              depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True, num_classes=555)
    # The rootpath of pretrained ViT
    checkpoint = torch.load(
        model_path, map_location='cpu')
    checkpoint = _conv_filter(checkpoint)
    model.load_state_dict(checkpoint, strict=True)
    return model


net = vit_base_patch16_384().to(device)

images_path = pd.read_csv(os.path.join(
    root_path, 'images.txt'), sep=' ', names=['img_id', 'filepath'])
image_class_labels = pd.read_csv(os.path.join(
    root_path, 'image_class_labels.txt'), sep=' ', names=['img_id', 'target'])


def get_continuous_class_map(class_labels):
    label_set = set(class_labels)
    return {k: i for i, k in enumerate(label_set)}


label_map = get_continuous_class_map(image_class_labels['target'])
train_test_split = pd.read_csv(os.path.join(
    root_path, 'train_test_split.txt'), sep=' ', names=['img_id', 'is_training_img'])

data = images_path.merge(image_class_labels, on='img_id')
data = data.merge(train_test_split, on='img_id')

data = data[data.is_training_img == 1]


def generate_heatmaps(data):
    for index in tqdm(range(len(data))):
        sample = data.iloc[index]
        path = sample.filepath
        pic_path = os.path.join(pic_root_path, path)
        save_path = os.path.join(heatmaps_path, path)

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
generate_heatmaps(data)
