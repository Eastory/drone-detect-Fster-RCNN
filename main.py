# main.py - 项目主执行入口
import os
from train import train_model
from infer import predict_drone
from config import CHECKPOINT_DIR, TEST_IMAGE_PATH

if __name__ == "__main__":
    # 选择执行模式："train" 或 "infer"
    mode = "train"  # 先训练，再改为"infer"

    if mode == "train":
        # 启动训练
        train_model()
    elif mode == "infer":
        # 启动推理（需先训练完成，替换为实际的权重文件）
        model_path = os.path.join(CHECKPOINT_DIR, "drone_fasterrcnn_epoch_10.pth")
        # 检查权重文件是否存在
        if not os.path.exists(model_path):
            print(f"错误：权重文件 {model_path} 不存在，请先训练模型！")
        else:
            predict_drone(TEST_IMAGE_PATH, model_path)
    else:
        print("错误：mode只能是'train'或'infer'！")