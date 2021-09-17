import random
import numpy as np
from PIL import Image


###### Function：Generate samples through three augmentation methods ######


def open_image(image_path):  # open the images
    image = Image.open(image_path)
    image = image.convert('RGB')
    return image


def open_heatmap(heat_path):  # open the heatmaps, return numpy
    heatmap = Image.open(heat_path)
    heatmap = np.array(heatmap)
    return heatmap


def cutout(image_path, heat_path):  # cutout
    image = open_image(image_path)

    if random.random() > 0.7:
        heatmap = open_heatmap(heat_path)

        p_threshold = np.random.randint(40, 60)/100.  # random thresholds
        pixel_threshold = heatmap.max()*p_threshold
        mask = heatmap > pixel_threshold
        mask = 1-mask  # final mask

        image = image.resize((438, 438))  # resize to the fixed size
        image = mask*image
        image = Image.fromarray(image.astype('uint8')).convert('RGB')

    return image


def enlargement(image_path, heat_path):  # enlargement
    image = open_image(image_path)

    if random.random() > 0.7:

        image = image.resize((438, 438))
        W, H = image.size  # width and height of the image

        heatmap = open_heatmap(heat_path)
        pixel_max = np.where(heatmap == np.max(
            heatmap))  # maximum of the heatmap

        # the point corresponding to the maximum of the hetmap
        max_point = (pixel_max[0][0], pixel_max[1][0])

        # choose the width and height of an area randomly
        W_ = np.random.randint(5, 15)/100.*W
        H_ = np.random.randint(5, 15)/100.*H

        enlarge_area = (max_point[0]-W_, max_point[1]-H_,
                        max_point[0]+W_, max_point[1]+H_)
        enlarge_image = image.crop(enlarge_area)

        W_a, H_a = enlarge_image.size  # width and height of the random area
        enlarge_image = enlarge_image.resize(
            (W_a*2, H_a*2))  # enlarge the subimage
        enlarge_area = (max_point[0]-W_a, max_point[1] -
                        H_a, max_point[0]+W_a, max_point[1]+H_a)  # area arter enlargement

        image.paste(enlarge_image, enlarge_area)

    return image


def flipping(image_path, heat_path):  # flipping
    image = open_image(image_path)

    if random.random() > 0.7:
        heatmap = open_heatmap(heat_path)
        pixel_max = np.where(heatmap == np.max(heatmap))
        max_point = (pixel_max[0][0], pixel_max[1][0])

        image = image.resize((438, 438))
        W, H = image.size

        W_ = np.random.randint(5, 15)/100.*W
        H_ = np.random.randint(5, 15)/100.*H

        flip_area = (max_point[0]-int(W_), max_point[1]-int(H_),
                     max_point[0]+int(W_), max_point[1]+int(H_))
        flip_image = image.crop(flip_area)
        flip_image = flip_image.transpose(
            Image.FLIP_LEFT_RIGHT)  # the subimage after flipping

        image.paste(flip_image, flip_area)

    return image


def augmentation(image_path, heat_path):
    return random.choice([enlargement, cutout, flipping])(image_path, heat_path)
