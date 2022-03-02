from torch.utils.data import Dataset
from PIL import Image
import random
from torchvision.transforms import Compose
from src.lib.augmentations import *
import pandas as pd


class ImageList(Dataset):

    def __init__(self, image_list, imsize=None, transform=None):
        Dataset.__init__(self)
        self.image_list = image_list
        self.transform = transform
        self.imsize = imsize

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, i):
        x = Image.open(self.image_list[i])
        x = x.convert("RGB")
        if self.transform is not None:
            x = self.transform(x)
        return x


class ImageList_with_label(Dataset):

    def __init__(self, image_list, labels, imsize=None, transform=None):
        Dataset.__init__(self)
        self.image_list = image_list
        self.labels = labels
        self.transform = transform
        self.imsize = imsize

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, i):
        label = self.labels[i]
        x = Image.open(self.image_list[i])
        x = x.convert("RGB")
        if self.transform is not None:
            x = self.transform(x)
        return x, label


class TripletTrainList(Dataset):

    def __init__(self, image_path, train_frame, imsize=None, transform=None, argumentation=None):
        Dataset.__init__(self)
        self.train_frame = train_frame
        self.image_path = image_path
        self.image_list = list(train_frame['path'])
        for j in range(len(self.image_list)):
            self.image_list[j] = image_path + self.image_list[j]
        self.transform = transform
        self.argumentation = argumentation
        self.imsize = imsize
        self.label_list = list(train_frame['MET_id'])
        self.frequencies = list(train_frame['class_frequency'])

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, i):
        # background = Image.open(random.sample(self.full_list, 1)[0])
        # self.argumentation.append(MergeImage(background, probability=0.3))
        # random.shuffle(self.argumentation)
        # argument = Compose(self.argumentation)
        label = self.train_frame['MET_id'][i]
        db_positive = Image.open(self.image_list[i])
        db_positive = db_positive.convert("RGB")
        if self.frequencies[i] == 1:
            query_image = self.argumentation(db_positive)
            sub_list = self.image_list.copy()
            sub_list.remove(self.image_list[i])
            db_negative = Image.open(random.choice(sub_list))
            db_negative = db_negative.convert("RGB")
        else:
            sub_f_p = self.train_frame[self.train_frame['MET_id'] == label]
            sub_f_n = self.train_frame[self.train_frame['MET_id'] != label]
            sub_list_p = list(sub_f_p['path'])
            for j in range(len(sub_list_p)):
                sub_list_p[j] = self.image_path + sub_list_p[j]
            sub_list_p.remove(self.image_list[i])
            sub_list_n = list(sub_f_n['path'])
            for j in range(len(sub_list_n)):
                sub_list_n[j] = self.image_path + sub_list_n[j]
            query_image = Image.open(random.choice(sub_list_p))
            db_negative = Image.open(random.choice(sub_list_n))
            query_image = query_image.convert("RGB")
            db_negative = db_negative.convert("RGB")
        if self.transform is not None:
            query_image = self.transform(query_image)
            db_positive = self.transform(db_positive)
            db_negative = self.transform(db_negative)
        return query_image, db_positive, db_negative


class TripletValList(Dataset):

    def __init__(self, image_list, imsize=None, transform=None, argumentation=None):
        Dataset.__init__(self)
        self.image_list = image_list
        self.transform = transform
        self.argumentation = argumentation
        self.imsize = imsize

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, i):
        random.shuffle(self.argumentation)
        argument = Compose(self.argumentation)
        q, r_p, r_n = self.image_list[i]
        query_image = Image.open(q)
        db_positive = Image.open(r_p)
        db_positive = argument(db_positive)
        db_negative = Image.open(r_n)
        query_image = query_image.convert("RGB")
        db_positive = db_positive.convert("RGB")
        db_negative = db_negative.convert("RGB")
        if self.transform is not None:
            query_image = self.transform(query_image)
            db_positive = self.transform(db_positive)
            db_negative = self.transform(db_negative)

        return query_image, db_positive, db_negative


class ContrastiveValList(Dataset):

    def __init__(self, image_list, imsize=None, transform=None):
        Dataset.__init__(self)
        self.image_list = image_list
        self.transform = transform
        self.imsize = imsize

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, i):
        q, r, label = self.image_list[i]
        query_image = Image.open(q)
        db_image = Image.open(r)
        query_image = query_image.convert("RGB")
        db_image = db_image.convert("RGB")
        if self.transform is not None:
            query_image = self.transform(query_image)
            db_image = self.transform(db_image)
        return query_image, db_image, label


class ContrastiveTrainList(Dataset):

    def __init__(self, image_list, full_list, imsize=None, transform=None, argumentation=None):
        Dataset.__init__(self)
        self.image_list = image_list
        self.transform = transform
        self.imsize = imsize
        self.argumentation = argumentation
        self.full_list = full_list

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, i):
        label = random.randint(0, 1)
        background = Image.open(random.sample(self.full_list, 1)[0])
        self.argumentation.append(MergeImage(background, probability=0.1))
        random.shuffle(self.argumentation)
        argument = Compose(self.argumentation)
        if label == 0:
            db_image = Image.open(self.image_list[i])
            db_image = db_image.convert("RGB")
            query_image = argument(db_image)
            if self.transform is not None:
                query_image = self.transform(query_image)
                db_image = self.transform(db_image)
        else:
            db_image = Image.open(self.image_list[i])
            query_image = Image.open(random.sample(self.full_list, 1)[0])
            query_image = query_image.convert("RGB")
            db_image = db_image.convert("RGB")
            if self.transform is not None:
                query_image = self.transform(query_image)
                db_image = self.transform(db_image)
        return query_image, db_image, label




