import os
import pickle
import pathlib
import torch
from torch.utils.data import Dataset
import torchvision
from PIL import Image
from pycocotools.coco import COCO
import pandas as pd
import torchvision.transforms as T

class HighwayDataset(Dataset):
    def __init__(self, root) -> None:
        super().__init__()
        self.root = root
        self.transform = T.PILToTensor()
        self.annotations = pd.read_csv(os.path.join(self.root, "annotations.csv"))

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        imgs = []

        anno = self.annotations.loc[index].to_dict()
        id = anno['id']

        for i in range(4):
            # (1, H, W)
            img = Image.open(os.path.join(self.root, str(id) + "_" + str(i) + ".png"))
            imgs.append(img.copy())

        if self.transform is not None:
            # (1, H, W)
            imgs = [self.transform(img) for img in imgs] 

        # (1, H, W * 4)
        imgs = torch.cat(imgs, dim=2)
        return imgs, anno

class DoomDataset(Dataset):
    def __init__(self, root) -> None:
        super().__init__()
        self.root = root
        self.annotations = pd.read_csv(os.path.join(self.root, "annotations.csv"))

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        doom_annotation = self.annotations.loc[index].to_dict()
        doom_id = doom_annotation['id']

        # {'scree': (5, 60, 40), 'variable': (10)}
        state = torch.load(os.path.join(self.root, str(doom_id) + ".pt"))

        return state, doom_annotation 

class AtariDataset(Dataset):
    def __init__(self, root) -> None:
        super().__init__()
        self.root = root
        self.transform = T.PILToTensor()
        self.annotations = pd.read_csv(os.path.join(self.root, "annotations.csv"))

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        imgs = []

        atari_annotation = self.annotations.loc[index].to_dict()
        atari_id = atari_annotation['id']

        for i in range(4):
            # (H, W)
            img = Image.open(os.path.join(self.root, str(atari_id) + "_" + str(i) + ".png")).convert("RGB")
            imgs.append(img.copy())

        if self.transform is not None:
            # (3, H, W)
            imgs = [self.transform(img) for img in imgs] 

        # (3, H, W * 4)
        imgs = torch.cat(imgs, dim=2)
        return imgs, atari_annotation

class MarioDataset(Dataset):
    def __init__(self, root, transform_fn=None) -> None:
        self.root = root
        self.transform = transform_fn
        self.annotations = pd.read_csv(os.path.join(self.root, "annotations.csv"))
    
    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        imgs = []

        mario_annotation = self.annotations.loc[index].to_dict()
        mario_id = mario_annotation['id']

        for i in range(4):
            # (3, H, W)
            img = Image.open(os.path.join(self.root, str(mario_id) + "_" + str(i) + ".png")).convert("RGB")
            imgs.append(img.copy())

        if self.transform is not None:
            imgs = [self.transform(img) for img in imgs] 
        # (3, H, W * 4)
        imgs = torch.cat(imgs, dim=2)
        return imgs, mario_annotation

class COCODataset(Dataset):
    def __init__(self, root, annotation, transform_fn=None):
        self.root = root
        self.transform = transform_fn
        self.coco = COCO(annotation)
        self.ids = list(sorted(self.coco.imgs.keys()))

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        coco_annotation = coco.loadAnns(ann_ids)
        path = coco.loadImgs(img_id)[0]['file_name']
        img = Image.open(os.path.join(self.root, path)).convert("RGB")

        num_objects = len(coco_annotation)
        cat_ids = []
        for i in range(num_objects):
            cat_ids.append(coco_annotation[i]['category_id'])

        targets = coco.getCatIds(catIds=cat_ids)

        my_annotation = {}
        my_annotation["targets"] = targets
        my_annotation["image_id"] = img_id
        my_annotation["filename"] = path

        if self.transform is not None:
            img = self.transform(img)

        return img, my_annotation

    def __len__(self):
        return len(self.ids)

class CUB200Dataset(Dataset):
    def __init__(self, root, annotations, transform_fn=None):
        self.root = root
        self.transform = transform_fn
        with open(annotations, 'rb') as fp:
            self.annotations = pickle.load(fp)

    def __getitem__(self, index):
        annotation = self.annotations[index]
        filename = pathlib.PurePath(annotation['filename']).name

        img = Image.open(os.path.join(self.root, filename)).convert("RGB")
        class_label = annotation['class']['label']

        my_annotation = {}
        my_annotation["target"] = class_label
        my_annotation["filename"] = filename

        if self.transform is not None:
            img = self.transform(img)

        return img, my_annotation

    def __len__(self):
        return len(self.annotations)

def main():
    ds = HighwayDataset("/home/rzuo02/work/visionmask_datasets/HIGHWAY/dqn/all")
    s = (ds[1000])[0]
    s = s.numpy()
    print(s.shape)
    img = Image.fromarray(s[0])
    img.save("foo.png")

if __name__ == "__main__":
    main()