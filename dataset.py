# dataset.py - 数据集加载与预处理
from PIL import Image
from datasets import load_dataset
from torch.utils.data import Dataset
from torchvision.transforms import functional as F
from config import DATASET_NAME, NUM_CLASSES, RANDOM_SEED, TRAIN_SAMPLE_NUM, TEST_SAMPLE_NUM
from utils import convert_coco_to_pascal
import torch
from torchvision.transforms import Compose, RandomHorizontalFlip, ColorJitter, RandomResizedCrop
import random
# ====================== 可自定义参数（新手直接改这里即可） ======================
# TRAIN_SAMPLE_NUM = 500  # 训练集只取500个随机样本（可改：1000/10000/20000等）
# RANDOM_SEED = 42  # 固定随机种子，确保每次取的样本一致（避免结果波动）

# 新增：定义检测任务的数据增强（图像+框同步变换）
def apply_augmentation(image, boxes):
    """
    对图像和边界框进行同步增强
    :param image: PIL图像
    :param boxes: 边界框（Pascal VOC格式: [xmin, ymin, xmax, ymax]）
    :return: 增强后的图像、同步变换后的框
    """
    # 1. 随机水平翻转（50%概率）
    if random.random() > 0.5:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
        # 框同步翻转（x轴镜像）
        width = image.width
        boxes = [[width - x2, y1, width - x1, y2] for [x1, y1, x2, y2] in boxes]

    # 2. 色域变换（亮度/对比度/饱和度）
    color_aug = ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
    image = color_aug(image)

    # 3. 随机缩放（保持宽高比）
    scale_factor = random.uniform(0.8, 1.2)
    new_width = int(image.width * scale_factor)
    new_height = int(image.height * scale_factor)
    image = image.resize((new_width, new_height), Image.BILINEAR)
    # 框同步缩放
    boxes = [[x1 * scale_factor, y1 * scale_factor, x2 * scale_factor, y2 * scale_factor] for [x1, y1, x2, y2] in
             boxes]

    return image, boxes


class DroneCOCODataset(Dataset):
    def __init__(self, dataset_split):
        """
        初始化无人机数据集（支持随机取训练集样本）
        :param dataset_split: 数据集拆分（"train" / "test"）
        """
        # 1. 加载原始数据集（使用本地缓存，无需重新下载）
        self.dataset = load_dataset(DATASET_NAME, split=dataset_split)

        # 2. 仅对训练集随机取N个样本（测试集保持完整）
        # 训练集数量51446，太多了，如果想完整训练，注释掉整个2即可
        if dataset_split == "train":
            print(f"⚠️  训练集原始样本数：{len(self.dataset)}")
            # 第一步：打乱训练集样本（固定seed保证可复现）
            self.dataset = self.dataset.shuffle(seed=RANDOM_SEED)
            # 第二步：取前TRAIN_SAMPLE_NUM个样本
            self.dataset = self.dataset.select(range(TRAIN_SAMPLE_NUM))
            print(f"✅ 训练集已随机选取 {TRAIN_SAMPLE_NUM} 个样本")

        # ====================== 新增：test集随机抽样 ======================
        # 如对训练集随机抽样一样，对测试集也进行随机抽样
        elif dataset_split == "test":
            print(f"⚠️  测试集原始样本数：{len(self.dataset)}")
            self.dataset = self.dataset.shuffle(seed=RANDOM_SEED)
            self.dataset = self.dataset.select(range(TEST_SAMPLE_NUM))
            print(f"✅ 测试集已随机选取 {TEST_SAMPLE_NUM} 个样本")

        # 3. 类别映射：数据集的0（无人机）→ 模型的1（模型0是背景）
        self.class2id = {0: 1}

    def __len__(self):
        return len(self.dataset)


    def __getitem__(self, idx):
        # 1. 加载单条样本
        sample = self.dataset[idx]
        image = sample['image'].convert('RGB')  # 确保RGB格式
        coco_bboxes = sample['objects']['bbox']
        categories = sample['objects']['category']

        # 2. 转换bbox格式（COCO → Pascal VOC）
        boxes = []
        for bbox in coco_bboxes:
            pascal_bbox = convert_coco_to_pascal(bbox)
            boxes.append(pascal_bbox)

        # ===== 新增：应用数据增强 =====
        image, boxes = apply_augmentation(image, boxes)
        # =============================

        # 3. 转换为张量（适配模型输入）
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor([self.class2id[c] for c in categories], dtype=torch.int64)

        # 4. 构建目标字典（符合torchvision检测模型要求）
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([idx]),
            'area': torch.as_tensor(sample['objects']['area'], dtype=torch.float32),
            'iscrowd': torch.zeros(len(boxes), dtype=torch.int64)
        }

        # 5. 图像转张量（归一化到[0,1]）
        img_tensor = F.to_tensor(image)

        return img_tensor, target



# 测试数据集加载（运行dataset.py时验证）
if __name__ == "__main__":
    train_dataset = DroneCOCODataset("train")
    img, target = train_dataset[0]
    print(f"图像张量形状：{img.shape}")
    print(f"边界框：{target['boxes']}")
    print(f"类别：{target['labels']}")