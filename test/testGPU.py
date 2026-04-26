import torch
# 检查是否支持CUDA
print("CUDA是否可用：", torch.cuda.is_available())
# 检查GPU数量
print("GPU数量：", torch.cuda.device_count())
# 显示GPU名称
if torch.cuda.is_available():
    print("GPU名称：", torch.cuda.get_device_name(0))