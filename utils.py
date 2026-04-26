# utils.py - 工具函数
import torch
from config import EPOCHS
import numpy as np

# COCO格式(x, y, w, h) → Pascal VOC格式(xmin, ymin, xmax, ymax)
def convert_coco_to_pascal(bbox):
    x, y, w, h = bbox
    xmin = x
    ymin = y
    xmax = x + w
    ymax = y + h
    return [xmin, ymin, xmax, ymax]

# 检测任务的collate_fn（解决单图目标数量不一致的问题）
def collate_fn(batch):
    return tuple(zip(*batch))

# 打印训练日志（格式化输出）
def print_train_log(epoch, epoch_loss, lr):
    log_str = f"Epoch [{epoch+1}/{EPOCHS}] | Avg Loss: {epoch_loss:.4f} | LR: {lr:.6f}"
    print("-" * 50)
    print(log_str)
    print("-" * 50)


# ====================== 新增：目标检测评估指标计算 ======================

def calculate_iou(box1, box2):
    """
    计算两个边界框的IoU（交并比）
    box1/box2格式：[xmin, ymin, xmax, ymax]
    """
    # 计算交集的坐标
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # 交集面积
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    if intersection == 0:
        return 0.0

    # 并集面积
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    # IoU = 交集/并集
    return intersection / union


def evaluate_model(model, test_loader, device, conf_threshold=0.5, iou_threshold=0.5):
    """
    用test集评估模型，计算Precision、Recall、F1-score
    :param model: 训练好的模型
    :param test_loader: test集的DataLoader
    :param device: 设备（cuda/cpu）
    :param conf_threshold: 预测置信度阈值（低于则过滤）
    :param iou_threshold: IoU阈值（≥则认为检测正确）
    :return: Precision, Recall, F1-score
    """
    model.eval()  # 切换到评估模式（禁用Dropout等）
    tp = 0  # 真阳性
    fp = 0  # 假阳性
    fn = 0  # 假阴性

    with torch.no_grad():  # 禁用梯度计算，节省显存
        for images, targets in test_loader:
            # 数据移到设备
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # 模型预测
            outputs = model(images)

            # 遍历每个样本的预测和真实标注
            for output, target in zip(outputs, targets):
                # 过滤低置信度的预测框
                pred_boxes = output['boxes'][output['scores'] >= conf_threshold].cpu().numpy()
                pred_scores = output['scores'][output['scores'] >= conf_threshold].cpu().numpy()

                # 真实框
                true_boxes = target['boxes'].cpu().numpy()
                true_labels = target['labels'].cpu().numpy()  # 1=无人机

                # 标记哪些真实框已被匹配
                true_matched = [False] * len(true_boxes)

                # 匹配预测框和真实框
                for pred_box in pred_boxes:
                    best_iou = 0.0
                    best_idx = -1
                    # 找和当前预测框IoU最大的真实框
                    for i, true_box in enumerate(true_boxes):
                        if not true_matched[i]:
                            iou = calculate_iou(pred_box, true_box)
                            if iou > best_iou:
                                best_iou = iou
                                best_idx = i
                    # 判断是否匹配成功
                    if best_iou >= iou_threshold:
                        tp += 1
                        true_matched[best_idx] = True  # 标记该真实框已匹配
                    else:
                        fp += 1

                # 未被匹配的真实框 = 漏检（FN）
                fn += sum([not matched for matched in true_matched])

    # 计算指标（避免分母为0）
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return precision, recall, f1