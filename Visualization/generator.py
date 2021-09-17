import cv2
import torch
import numpy as np
import torchvision.transforms as transforms

from PIL import Image
from Visualization.ViT_explanation_generator import LRP

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
transform = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
    normalize,
])


def open_image(image_path):
    image = Image.open(image_path)
    image = image.convert('RGB')
    return image


def generate_heatmaps(attribution_generator, original_image, class_index=None):
    transformer_attribution = attribution_generator.generate_LRP(original_image.unsqueeze(
        0).to(device), method="transformer_attribution", index=class_index).detach()
    transformer_attribution = transformer_attribution.reshape(1, 1, 24, 24)
    transformer_attribution = torch.nn.functional.interpolate(
        transformer_attribution, scale_factor=16, mode='bilinear', align_corners=True)
    transformer_attribution = transformer_attribution.reshape(
        384, 384).to(device).data.cpu().numpy()
    transformer_attribution = (transformer_attribution - transformer_attribution.min()) / (
        transformer_attribution.max() - transformer_attribution.min())
    map = np.uint8(255 * transformer_attribution)
    map = cv2.cvtColor(np.array(map), cv2.COLOR_RGB2BGR)
    return map


def image2heatmap(net, image_path):  # Require net and all images' path

    attribution_generator = LRP(net)

    image = open_image(image_path)
    image = transform(image)
    heatmap = generate_heatmaps(attribution_generator, image)
    return heatmap
