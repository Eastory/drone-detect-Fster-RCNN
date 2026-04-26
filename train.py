# train.py - 模型训练逻辑
import torch
import os
from torch.utils.data import DataLoader
from tqdm import tqdm
from config import EPOCHS, BATCH_SIZE, LEARNING_RATE, MOMENTUM, WEIGHT_DECAY, STEP_SIZE, GAMMA, DEVICE, CHECKPOINT_DIR, \
    LR_SCHEDULER_TYPE, T_MAX, ETA_MIN
from dataset import DroneCOCODataset
from model import build_fasterrcnn_model
from utils import collate_fn, print_train_log, evaluate_model


def train_model():
    """
    训练Faster R-CNN模型（新增每轮评估逻辑）
    :return: 训练好的模型
    """
    # 1. 加载数据集
    print("加载训练/测试数据集...")
    train_dataset = DroneCOCODataset("train")
    val_dataset = DroneCOCODataset("test")  # 用test集作为评估集

    # 2. 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0  # Windows设为0
    )
    # 新增：创建test集的DataLoader（评估用）
    test_loader = DataLoader(
        val_dataset,
        batch_size=1,  # 评估时批次设为1，避免匹配错误
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )

    # 3. 构建模型
    print("构建Faster R-CNN模型...")
    model = build_fasterrcnn_model()

    # 4. 定义优化器和学习率调度器
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=LEARNING_RATE,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY
    )
    # train.py 中学习率调度器部分
    if LR_SCHEDULER_TYPE == "CosineAnnealing":
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=T_MAX,
            eta_min=ETA_MIN
        )
    else:  # 保留StepLR
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=STEP_SIZE,
            gamma=GAMMA
        )

    # 5. 开始训练
    print(f"开始训练（共{EPOCHS}轮）...")
    model.train()
    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}")

        for images, targets in pbar:
            # 将数据移到指定设备
            images = [img.to(DEVICE) for img in images]
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

            # 前向传播计算损失
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            # 反向传播+优化
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            # 累计损失
            epoch_loss += losses.item()
            pbar.set_postfix({"batch_loss": losses.item()})

        # 更新学习率
        lr_scheduler.step()

        # 打印本轮训练日志
        avg_loss = epoch_loss / len(train_loader)
        current_lr = optimizer.param_groups[0]['lr']
        print_train_log(epoch, avg_loss, current_lr)

        # ====================== 新增：每轮训练后评估test集 ======================
        print(f"\n📊 开始评估test集（Epoch {epoch + 1}）...")
        precision, recall, f1 = evaluate_model(model, test_loader, DEVICE)
        print(f"✅ Epoch {epoch + 1} 评估结果：")
        print(f"   准确率（Precision）：{precision:.4f}")
        print(f"   召回率（Recall）：{recall:.4f}")
        print(f"   F1-score：{f1:.4f}")
        # =====================================================================

        # ====================== 新增：评估后切回训练模式 ======================
        model.train()  # 关键！评估后必须切回训练模式，否则第二轮训练报错
        # =====================================================================

        # 保存本轮模型权重
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f"drone_fasterrcnn_epoch_{epoch + 1}.pth")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"💾 模型权重已保存至：{checkpoint_path}\n")

    print("训练完成！")
    return model

# 单独运行train.py可直接启动训练
if __name__ == "__main__":
    train_model()
