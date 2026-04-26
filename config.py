# config.py - 全局配置参数
import torch

# ====================== 设备配置 ======================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备：{DEVICE}")

# ====================== 数据集配置 ======================
DATASET_NAME = "pathikg/drone-detection-dataset"  # Hugging Face数据集名称
TRAIN_SAMPLE_NUM = 500  # 训练集随机抽样数量
TEST_SAMPLE_NUM = 300   # test集随机抽样数量（可改：100/500/1000等）
RANDOM_SEED = 42        # 固定随机种子，确保抽样可复现
NUM_CLASSES = 2  # 背景(0) + 无人机(1)

# ====================== 训练配置 ======================
EPOCHS = 10  # 训练轮数
BATCH_SIZE = 2  # 批次大小
LEARNING_RATE = 0.005  # 初始学习率
WEIGHT_DECAY = 0.0005  # 权重衰减（防止过拟合）
MOMENTUM = 0.9  # SGD优化器动量
STEP_SIZE = 3  # 学习率调度器步长（每3轮降一次学习率）
GAMMA = 0.1  # 学习率衰减系数（每次降为原来的10%）

# ====================== 推理配置 ======================
CONF_THRESHOLD = 0.7  # 检测置信度阈值（低于则过滤）
IMAGE_NAME = "499"
TEST_IMAGE_PATH = "testImages/input/"+ IMAGE_NAME + ".jpg"  # 测试图像路径（需替换为你的图片）
OUTPUT_IMAGE_PATH = "testImages/output/"+ IMAGE_NAME + "_result.jpg"  # 检测结果保存路径

# ====================== 路径配置 ======================
CHECKPOINT_DIR = "runs"  # 模型权重保存目录

LR_SCHEDULER_TYPE = "CosineAnnealing"  # StepLR/CosineAnnealing
T_MAX = EPOCHS  # CosineAnnealing的周期（等于训练轮数）
ETA_MIN = 1e-6  # CosineAnnealing的最小学习率

ANCHOR_SIZES = [32, 64, 128, 256]  # 无人机常见尺度（像素）
ANCHOR_ASPECT_RATIOS = [0.5, 1.0, 2.0]  # 无人机宽高比

