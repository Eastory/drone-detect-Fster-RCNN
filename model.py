# model.py - Faster R-CNN模型定义
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from config import NUM_CLASSES, DEVICE


def build_fasterrcnn_model():
    """
    构建适配无人机检测的Faster R-CNN模型
    :return: 初始化后的模型
    """
    # 加载预训练的Faster R-CNN基础模型（基于COCO数据集）
    model = fasterrcnn_resnet50_fpn(weights="FasterRCNN_ResNet50_FPN_Weights.COCO_V1")

    # 获取分类头的输入特征维度
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # 替换分类头（适配无人机数据集的类别数：背景+无人机=2类）
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)

    # 将模型移到指定设备（GPU/CPU）
    model = model.to(DEVICE)

    return model


# 测试模型构建（运行model.py时验证）
if __name__ == "__main__":
    model = build_fasterrcnn_model()
    print("模型构建成功！")
    print(f"模型设备：{next(model.parameters()).device}")