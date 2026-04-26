# infer.py - 模型推理与可视化

import os
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from torchvision.transforms import functional as F
from config import DEVICE, CONF_THRESHOLD, TEST_IMAGE_PATH, OUTPUT_IMAGE_PATH, NUM_CLASSES, CHECKPOINT_DIR
from model import build_fasterrcnn_model


def predict_drone(image_path, model_path, conf_threshold=CONF_THRESHOLD):
    """
    无人机检测推理
    :param image_path: 测试图像路径
    :param model_path: 训练好的模型权重路径
    :param conf_threshold: 置信度阈值
    """
    # 1. 加载模型并加载权重
    model = build_fasterrcnn_model()
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()  # 推理模式

    # 2. 加载并预处理图像
    image = Image.open(image_path).convert('RGB')
    img_tensor = F.to_tensor(image).to(DEVICE)
    inputs = [img_tensor]

    # 3. 推理（禁用梯度计算）
    with torch.no_grad():
        outputs = model(inputs)

    # 4. 解析检测结果
    detections = outputs[0]
    boxes = detections['boxes'][detections['scores'] >= conf_threshold].cpu().numpy()
    scores = detections['scores'][detections['scores'] >= conf_threshold].cpu().numpy()

    # 5. 可视化结果
    fig, ax = plt.subplots(1, figsize=(10, 8))
    ax.imshow(image)

    # 绘制边界框和置信度
    for box, score in zip(boxes, scores):
        x1, y1, x2, y2 = box
        # 绘制红色边界框
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2, edgecolor='red', facecolor='none'
        )
        ax.add_patch(rect)
        # 添加置信度标签
        ax.text(
            x1, y1 - 10, f"Drone ({score:.2f})",
            color='red', fontsize=12, weight='bold'
        )

    plt.axis('off')
    plt.tight_layout()
    plt.savefig(OUTPUT_IMAGE_PATH, dpi=150)
    plt.show()

    # 打印检测结果
    print(f"检测完成！结果已保存至：{OUTPUT_IMAGE_PATH}")
    print(f"检测到 {len(boxes)} 架无人机（置信度≥{conf_threshold}）")


# 单独运行infer.py可直接启动推理（需先替换model_path为实际权重路径）
if __name__ == "__main__":
    # 替换为你训练好的权重路径（比如runs/drone_fasterrcnn_epoch_10.pth）
    model_path = os.path.join(CHECKPOINT_DIR, "drone_fasterrcnn_epoch_10.pth")
    predict_drone(TEST_IMAGE_PATH, model_path)