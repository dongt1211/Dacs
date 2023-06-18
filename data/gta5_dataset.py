
from glob import glob
import cv2
import time
import os
import os.path as osp
import numpy as np
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import collections
import torch
import torchvision
from torch.utils import data
from PIL import Image
import json
def convert_to_train_id(file):
    # re-assign labels to match the format of Cityscapes
    pil_label = Image.open(file)
    label = np.asarray(pil_label)
    id_to_trainid = {
        7: 0,
        8: 1,
        11: 2,
        12: 3,
        13: 4,
        17: 5,
        19: 6,
        20: 7,
        21: 8,
        22: 9,
        23: 10,
        24: 11,
        25: 12,
        26: 13,
        27: 14,
        28: 15,
        31: 16,
        32: 17,
        33: 18
    }
    label_copy = 255 * np.ones(label.shape, dtype=np.uint8)
    sample_class_stats = {}
    for k, v in id_to_trainid.items():
        k_mask = label == k
        label_copy[k_mask] = v
        n = int(np.sum(k_mask))
        if n > 0:
            sample_class_stats[v] = n
    new_file = file.replace('.png', '_labelTrainIds.png')
    assert file != new_file
    sample_class_stats['file'] = new_file
    Image.fromarray(label_copy, mode='L').save(new_file)
    return sample_class_stats
def save_class_stats(out_dir, sample_class_stats):
    with open(osp.join(out_dir, 'sample_class_stats.json'), 'w') as of:
        json.dump(sample_class_stats, of, indent=2)

    sample_class_stats_dict = {}
    for stats in sample_class_stats:
        f = stats.pop('file')
        sample_class_stats_dict[f] = stats
    with open(osp.join(out_dir, 'sample_class_stats_dict.json'), 'w') as of:
        json.dump(sample_class_stats_dict, of, indent=2)

    samples_with_class = {}
    for file, stats in sample_class_stats_dict.items():
        for c, n in stats.items():
            if c not in samples_with_class:
                samples_with_class[c] = [(file, n)]
            else:
                samples_with_class[c].append((file, n))
    with open(osp.join(out_dir, 'samples_with_class.json'), 'w') as of:
        json.dump(samples_with_class, of, indent=2)
class GTA5DataSet(data.Dataset):
    def __init__(self, root, max_iters=None, augmentations = None, img_size=(321, 321), mean=(128, 128, 128), scale=True, mirror=True, ignore_label=250, used_images=None):
        self.root = root
        # self.list_path = list_path
        self.img_size = img_size
        self.scale = scale
        self.ignore_label = ignore_label
        self.mean = mean
        self.is_mirror = mirror
        self.augmentations = augmentations
        # self.mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])
        # self.img_ids = [i_id.strip() for i_id in open(list_path)]
        if not max_iters==None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.files = []

        self.id_to_trainid = {7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5,
                              19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12,
                              26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18}

        # # for split in ["train", "trainval", "val"]:
        # for name in self.img_ids:
        #     img_file = osp.join(self.root, "images/%s" % name)
        #     label_file = osp.join(self.root, "labels/%s" % name)
        #     self.files.append({
        #         "img": img_file,
        #         "label": label_file,
        #         "name": name
        #     })
        for img_path in glob(self.root + "/images/*.png"):
            mask_path = img_path.replace("images", "labels")
            name = img_path.split("/")[-1].replace(".png", "")
            if (used_images is not None) and (name in used_images):
                continue
            #Error pictures inside GTA5 dataset
            if(('/images/15188' not in img_path) and  ('/images/17705' not in img_path)):
                self.files.append(
                    {
                        "img": img_path,
                        "label": mask_path,
                        "name": name
                    }
            )

    def __len__(self):
        return len(self.files)


    def __getitem__(self, index):
        datafiles = self.files[index]

        image = Image.open(datafiles["img"]).convert('RGB')
        label = Image.open(datafiles["label"])
        name = datafiles["name"]

        # resize
        image = image.resize(self.img_size, Image.BICUBIC)
        label = label.resize(self.img_size, Image.NEAREST)

        image = np.asarray(image, np.uint8)
        label = np.asarray(label, np.uint8)

        if self.augmentations is not None:
            image, label = self.augmentations(image, label)

        image = np.asarray(image, np.float32)
        label = np.asarray(label, np.float32)

        # re-assign labels to match the format of Cityscapes
        label_copy = 255 * np.ones(label.shape, dtype=np.float32)
        for k, v in self.id_to_trainid.items():
            label_copy[label == k] = v

        size = image.shape
        image = image[:, :, ::-1]  # change to BGR
        image -= self.mean
        image = image.transpose((2, 0, 1))

        return image.copy(), label_copy.copy(), np.array(size), name


# if __name__ == '__main__':
#     dst = GTA5DataSet(root = "/home/admin_mcn/taint/dataset/gta5")
#     trainloader = data.DataLoader(dst, batch_size=4)
#     for i, data in enumerate(trainloader):
#         imgs, labels,_,_ = data
#         # img = torchvision.utils.make_grid(imgs).numpy()
#         # img = np.transpose(img, (1, 2, 0))
#         # img = img[:, :, ::-1]
#         print(imgs.shape)
if __name__ == '__main__':
    gta_path = '/kaggle/input/gtav-dataset/GTAV'
    out_dir = '/kaggle/input/gtav-dataset/GTAV'

    gt_dir = osp.join(gta_path, 'labels')

    import os

    def get_files_with_suffix(directory, suffixes, recursive=False):
      file_list = []
      for root, dirs, files in os.walk(directory):
        if not recursive and root != directory:
            continue
        for file in files:
            if file.endswith(suffixes):
                file_path = os.path.join(root, file)
                file_list.append(file_path)
      return sorted(file_list)


    suffixes = tuple(f'{i}.png' for i in range(10))
    poly_files = get_files_with_suffix(gt_dir, suffixes, recursive=True)
    only_postprocessing = False
    if not only_postprocessing:
            # sample_class_stats = mmcv.track_progress(convert_to_train_id,
            #                                          poly_files)
            converted_files = []
            for poly_file in tqdm(poly_files, desc="Converting files"):
              converted_file = convert_to_train_id(poly_file)
              converted_files.append(converted_file)
    else:
        with open(osp.join(out_dir, 'sample_class_stats.json'), 'r') as of:
            converted_files = json.load(of)

    save_class_stats(out_dir, converted_files)
