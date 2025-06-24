# 存放红外图像和标签（用于语义分割）

import os
from PIL import Image
from torch.utils.data import Dataset


class MFNetDataset(Dataset):
    def __init__(self, root, transform=None, max_samples=None):
        self.root = root
        self.transform = transform
        # 获取图片和标签文件名列表
        self.image_dir = os.path.join(root, 'images')
        self.label_dir = os.path.join(root, 'labels')
        self.img_names = sorted(os.listdir(self.image_dir))
        self.label_names = sorted(os.listdir(self.label_dir))
        # 只保留两者都存在的文件
        self.img_names = [f for f in self.img_names if f in self.label_names]
        self.label_names = [f for f in self.img_names]  # 保证顺序一致
        if max_samples is not None:
            self.img_names = self.img_names[:max_samples]
            self.label_names = self.label_names[:max_samples]

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.img_names[idx])
        label_path = os.path.join(self.label_dir, self.label_names[idx])
        # 红外图像通常为单通道，这里用L模式
        rgb = Image.open(img_path).convert('L')
        label = Image.open(label_path)
        if self.transform:
            rgb = self.transform(rgb)
            label = self.transform(label)
        return {'rgb': rgb, 'label': label}

    def __len__(self):
        return len(self.img_names)