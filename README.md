```DroneDetection_FasterRCNN/
├── test               # 放用来测试的代码
    ├── testData.py    # 测试数据集能不能正常用
    └── testGPU.py     # 测试 GPU 能不能用
├── testImages         # 测试图片
    ├── input
    └── output
├── config.py          # 全局配置参数（数据集、训练、设备等）
├── dataset.py         # 数据集加载+预处理（适配COCO格式bbox转换）
├── model.py           # Faster R-CNN模型定义（适配无人机单类别）
├── train.py           # 训练核心逻辑（优化器、损失计算、权重保存）
├── infer.py           # 推理+可视化（加载权重检测单张图片）
├── utils.py           # 工具函数（bbox转换、collate_fn等）
├── main.py            # 主执行入口（一键启动训练/推理）
└── runs/              # 自动保存训练权重，每一轮的训练都会保存对应的训练权重，测试或推断时需要使用对应的权重
```
训练（train）：
每一轮训练后会测试，计算准确率，召回率，F1这三个，实验结果主要就看这三个

推断（infer）：
拿一张图片根据某一轮的训练权重让他识别出无人机，输入图片放在input，输出图片放在output。
图片名字和所要使用的权重在config里面改

如何执行train或infer？
在main.py里面更改mode值
