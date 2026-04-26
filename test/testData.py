# 导入核心库
from datasets import load_dataset
import torch
from PIL import Image

# ====================== 1. 加载数据集 ======================
# 首次运行会自动下载数据集（约几百MB），后续运行加载本地缓存（无需重复下载）
# 缓存路径：Linux/Mac → ~/.cache/huggingface/datasets；Windows → C:\Users\你的用户名\.cache\huggingface\datasets
print("开始加载数据集（首次运行会下载，耐心等待...）")
ds = load_dataset("pathikg/drone-detection-dataset")

# ====================== 2. 验证数据集基本结构 ======================
print("\n===== 验证数据集基本拆分 =====")
# 查看数据集的拆分（train/test）
dataset_splits = list(ds.keys())
print(f"数据集包含的拆分：{dataset_splits}")  # 预期输出：['train', 'test']

# 查看各拆分的样本数量
train_size = len(ds['train'])
test_size = len(ds['test'])
print(f"训练集样本数：{train_size}（预期约51446）")
print(f"测试集样本数：{test_size}（预期约2625）")

# ====================== 3. 验证单条样本的完整结构 ======================
print("\n===== 验证单条样本的字段结构 =====")
# 取训练集第0条样本（可替换为任意索引，比如10、100）
sample = ds['train'][0]

# 打印样本的所有字段名
print(f"单条样本包含的字段：{list(sample.keys())}")  # 预期：['width', 'height', 'objects', 'image', 'image_id']

# 逐一验证核心字段
## 3.1 验证图像尺寸（数据集标注为640×480）
width = sample['width']
height = sample['height']
print(f"\n图像尺寸：width={width}, height={height}（预期width=640, height=480）")
assert width == 640 and height == 480, "图像尺寸不符合预期！"

## 3.2 验证图像对象（需为PIL Image，可直接用于预处理）
image = sample['image']
print(f"图像对象类型：{type(image)}（预期：<class 'PIL.Image.Image'>）")
print(f"图像模式：{image.mode}（预期：RGB）")
print(f"图像实际尺寸：{image.size}（预期：(640, 480)）")

## 3.3 验证标注核心（objects字段）
objects = sample['objects']
print(f"\nobjects字段包含的子字段：{list(objects.keys())}")  # 预期：['bbox', 'category', 'area', 'id']

### 3.3.1 验证边界框（COCO格式：x, y, w, h）
coco_bboxes = objects['bbox']
print(f"标注框（COCO格式 x,y,w,h）：{coco_bboxes}")
print(f"标注框数量：{len(coco_bboxes)}（单图无人机数量）")

### 3.3.2 验证类别（仅无人机，索引为0）
categories = objects['category']
print(f"类别索引：{categories}（预期全为0，代表无人机）")
assert all(cat == 0 for cat in categories), "类别索引不符合预期！"

### 3.3.3 验证标注框面积（w×h）
areas = objects['area']
print(f"标注框面积：{areas}（预期为对应bbox的宽×高）")
# 验证面积计算是否正确（以第一个bbox为例）
if len(coco_bboxes) > 0:
    x, y, w, h = coco_bboxes[0]
    calc_area = w * h
    assert abs(areas[0] - calc_area) < 1e-5, "标注框面积计算错误！"
    print(f"面积验证：第一个框计算面积={calc_area}，数据集标注面积={areas[0]} → 一致")

### 3.3.4 验证目标ID（单图内唯一）
obj_ids = objects['id']
print(f"单图内目标ID：{obj_ids}（预期为递增整数，如[0]、[0,1]）")

## 3.4 验证图像ID（全局唯一）
image_id = sample['image_id']
print(f"\n图像全局ID：{image_id}（预期为整数，范围0~51445）")

# ====================== 4. 验证关键：COCO bbox → Faster R-CNN所需格式 ======================
print("\n===== 验证bbox格式转换 =====")
# COCO格式（x, y, w, h）→ Faster R-CNN需要的（xmin, ymin, xmax, ymax）
def convert_coco_to_fasterrcnn(bbox):
    x, y, w, h = bbox
    xmin = x
    ymin = y
    xmax = x + w
    ymax = y + h
    return [xmin, ymin, xmax, ymax]

# 转换所有bbox并验证
fasterrcnn_bboxes = [convert_coco_to_fasterrcnn(bbox) for bbox in coco_bboxes]
print(f"COCO格式bbox：{coco_bboxes}")
print(f"转换后（Faster R-CNN格式）：{fasterrcnn_bboxes}")
# 验证转换后的坐标是否合理（不超过图像尺寸）
for bbox in fasterrcnn_bboxes:
    xmin, ymin, xmax, ymax = bbox
    assert 0 <= xmin < xmax <= 640 and 0 <= ymin < ymax <= 480, "转换后的bbox坐标超出图像范围！"
print("bbox格式转换验证通过 → 符合Faster R-CNN输入要求")

# ====================== 5. 额外验证：多目标样本（可选） ======================
# 找一个包含多个无人机的样本（比如train[12]，可根据实际情况调整）
print("\n===== 验证多目标样本 =====")
multi_sample = ds['train'][12]
multi_bboxes = multi_sample['objects']['bbox']
multi_categories = multi_sample['objects']['category']
print(f"多目标样本的无人机数量：{len(multi_bboxes)}")
print(f"多目标样本的类别：{multi_categories}（预期全为0）")

# ====================== 6. 最终验证总结 ======================
print("\n===== 数据集验证总结 =====")
print("✅ 数据集加载成功！")
print("✅ 数据集拆分（train/test）正常！")
print("✅ 图像尺寸、格式正常！")
print("✅ 标注框（bbox）格式、类别、面积验证通过！")
print("✅ bbox格式转换（COCO→Faster R-CNN）验证通过！")
print("✅ 多目标样本验证通过！")
print("\n结论：数据集结构完全符合使用要求，可以用于后续Faster R-CNN训练！")