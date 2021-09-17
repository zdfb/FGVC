import os
import sys
import scipy.io
import pandas as pd
from PIL import Image
from os.path import join
from torchvision import transforms
from torch.utils.data import Dataset
from torchvision.datasets.utils import list_dir
from utils.Augmentation import augmentation

###### Function: Load the image samples after augmentation  ######


class CUB_200(Dataset):
    def __init__(self, root, train=True, transform=None):
        super(CUB_200, self).__init__()
        self.root = root
        self.train = train
        self.transform_ = transform
        self.classes_file = os.path.join(
            root, 'classes.txt')  # <class_id> <class_name>
        self.image_class_labels_file = os.path.join(
            root, 'image_class_labels.txt')  # <image_id> <class_id>
        self.images_file = os.path.join(
            root, 'images.txt')  # <image_id> <image_name>
        self.train_test_split_file = os.path.join(
            root, 'train_test_split.txt')  # <image_id> <is_training_image>

        self._train_ids = []
        self._test_ids = []
        self._image_id_label = {}
        self._train_path_label = []
        self._test_path_label = []

        self._train_test_split()
        self._get_id_to_label()
        self._get_path_label()

    def _train_test_split(self):

        for line in open(self.train_test_split_file):
            image_id, label = line.strip('\n').split()
            if label == '1':
                self._train_ids.append(image_id)
            elif label == '0':
                self._test_ids.append(image_id)
            else:
                raise Exception('label error')

    def _get_id_to_label(self):
        for line in open(self.image_class_labels_file):
            image_id, class_id = line.strip('\n').split()
            self._image_id_label[image_id] = class_id

    def _get_path_label(self):
        for line in open(self.images_file):
            image_id, image_name = line.strip('\n').split()
            label = self._image_id_label[image_id]
            if image_id in self._train_ids:
                self._train_path_label.append((image_name, label))
            else:
                self._test_path_label.append((image_name, label))

    def __getitem__(self, index):

        if self.train:
            image_name, label = self._train_path_label[index]
            image_path = os.path.join(self.root, 'images', image_name)
            heat_path = os.path.join(self.root, 'heatmaps', image_name)
            img = augmentation(image_path, heat_path)

        else:
            image_name, label = self._test_path_label[index]
            image_path = os.path.join(self.root, 'images', image_name)
            img = Image.open(image_path)
            img = img.convert('RGB')

        label = int(label) - 1
        if self.transform_ is not None:
            img = self.transform_(img)
        return img, label

    def __len__(self):
        if self.train:
            return len(self._train_ids)

        else:
            return len(self._test_ids)


class StanfordDogs(Dataset):
    def __init__(self, root, train=True, transform=None):
        super(StanfordDogs, self).__init__()
        self.root = root
        self.train = train
        self.transform = transform
        split = self.load_split()

        self.images_folder = join(self.root, 'Images')
        self.heatmaps_folder = join(self.root, 'heatmaps')
        self._breeds = list_dir(self.images_folder)
        self._breed_images = [(annotation + '.jpg', idx)
                              for annotation, idx in split]
        self._flat_breed_images = self._breed_images

        self._breed_images = [(annotation + '.jpg', idx)
                              for annotation, idx in split]

        self._flat_breed_images = self._breed_images

    def __len__(self):
        return len(self._flat_breed_images)

    def __getitem__(self, index):
        image_name, target = self._flat_breed_images[index]
        image_path = join(self.images_folder, image_name)

        if self.train:
            heatmap_path = join(self.heatmaps_folder, image_name)
            image = augmentation(image_path, heatmap_path)

        else:
            image = Image.open(image_path)
            image = image.convert('RGB')

        if self.transform is not None:
            image = self.transform(image)
        return image, target

    def load_split(self):
        if self.train:
            split = scipy.io.loadmat(join(self.root, 'train_list.mat'))[
                'annotation_list']
            labels = scipy.io.loadmat(
                join(self.root, 'train_list.mat'))['labels']
        else:
            split = scipy.io.loadmat(join(self.root, 'test_list.mat'))[
                'annotation_list']
            labels = scipy.io.loadmat(
                join(self.root, 'test_list.mat'))['labels']

        split = [item[0][0] for item in split]
        labels = [item[0] - 1 for item in labels]
        return list(zip(split, labels))


class StanfordCars(Dataset):
    def __init__(self, root, train=True, transform=None):
        super(StanfordCars, self).__init__()
        self.root = root
        self.train = train
        self.transform = transform
        self.heatmaps_folder = join(self.root, 'heatmaps')

        loaded_mat = scipy.io.loadmat(join(self.root, 'cars_annos.mat'))
        loaded_mat = loaded_mat['annotations'][0]

        self.samples = []
        for item in loaded_mat:
            if self.train != bool(item[-1][0]):
                path = str(item[0][0])
                label = int(item[-2][0]) - 1
                self.samples.append((path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, target = self.samples[index]

        if self.train:
            image_path = join(self.root, path)
            hearmap_path = join(self.heatmaps_folder, path[8:])
            image = augmentation(image_path, hearmap_path)

        else:
            image_path = join(self.root, path)
            image = Image.open(image_path)
            image = image.convert('RGB')

        if self.transform is not None:
            image = self.transform(image)

        return image, target


class OxfordFlowers(Dataset):
    def __init__(self, root, train=True, transform=None):
        super(OxfordFlowers, self).__init__()
        self.root = root
        self.train = train
        self.transform = transform

        self.images_path = join(self.root, 'jpg')
        self.heatmaps_path = join(self.root, 'heatmaps')
        labes_path = join(self.root, 'imagelabels.mat')
        split_path = join(self.root, 'setid.mat')

        images_path = os.listdir(self.images_path)
        labels = scipy.io.loadmat(labes_path)
        labels = labels['labels'][0] - 1
        split = scipy.io.loadmat(split_path)

        if self.train:
            image_index = split['trnid'][0]-1
        else:
            image_index = split['tstid'][0]-1

        self.samples = []
        for item in image_index:
            path = images_path[item]
            label = labels[item]
            self.samples.append((path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, target = self.samples[index]
        image_path = join(self.images_path, path)

        if self.train:
            heatmap_path = join(self.heatmaps_path, path)
            image = augmentation(image_path, heatmap_path)

        else:
            image = Image.open(image_path)
            image = image.convert('RGB')

        if self.transform is not None:
            image = self.transform(image)

        return image, target


class NABirds(Dataset):
    def __init__(self, root, train=True, transform=None):
        super(NABirds, self).__init__()
        self.root = root
        self.train = train
        self.transform = transform

        images_path = pd.read_csv(os.path.join(
            self.root, 'images.txt'), sep=' ', names=['img_id', 'filepath'])
        image_class_labels = pd.read_csv(os.path.join(
            self.root, 'image_class_labels.txt'), sep=' ', names=['img_id', 'target'])

        self.label_map = get_continuous_class_map(image_class_labels['target'])
        train_test_split = pd.read_csv(os.path.join(
            self.root, 'train_test_split.txt'), sep=' ', names=['img_id', 'is_training_img'])

        data = images_path.merge(image_class_labels, on='img_id')
        self.data = data.merge(train_test_split, on='img_id')

        if self.train:
            self.data = self.data[self.data.is_training_img == 1]
        else:
            self.data = self.data[self.data.is_training_img == 0]

        self.class_names = load_class_names(self.root)
        self.class_hierarchy = load_hierarchy(self.root)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]

        if self.train:
            path = os.path.join(self.root, 'images', sample.filepath)
            heat_path = os.path.join(self.root, 'heatmaps', sample.filepath)
            img = augmentation(path, heat_path)
        else:
            path = os.path.join(self.root, 'images', sample.filepath)
            img = Image.open(path)
            img = img.convert('RGB')

        target = self.label_map[sample.target]
        if self.transform is not None:
            img = self.transform(img)
        return img, target


def get_continuous_class_map(class_labels):
    label_set = set(class_labels)
    return {k: i for i, k in enumerate(label_set)}


def load_class_names(dataset_path):
    names = {}

    with open(os.path.join(dataset_path, 'classes.txt')) as f:
        for line in f:
            pieces = line.strip().split()
            class_id = pieces[0]
            names[class_id] = ' '.join(pieces[1:])

    return names


def load_hierarchy(dataset_path):
    parents = {}

    with open(os.path.join(dataset_path, 'hierarchy.txt')) as f:
        for line in f:
            pieces = line.strip().split()
            child_id, parent_id = pieces
            parents[child_id] = parent_id

    return parents


data_transform = {
    "train": transforms.Compose([
        transforms.Resize(438),
        transforms.RandomResizedCrop(384),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]),
    "test": transforms.Compose([transforms.Resize(438),
                               transforms.CenterCrop(384),
                               transforms.ToTensor(),
                               transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])}


def load_datasets(dataset, root, train, transform):  # choose the dataset

    if dataset == 'CUB_200_2011':
        return CUB_200(root, train, transform)
    elif dataset == 'NABirds':
        return NABirds(root, train, transform)
    elif dataset == 'Cars':
        return StanfordCars(root, train, transform)
    elif dataset == 'Dogs':
        return StanfordDogs(root, train, transform)
    elif dataset == 'Flowers':
        return OxfordFlowers(root, train, transform)
