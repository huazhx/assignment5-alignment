import os
from configs.config import settings

# 1. 在导入 torch 之前设置 GPU 环境
# os.environ["CUDA_VISIBLE_DEVICES"] = settings.cuda_visible_devices

# 2. 导入深度学习库
import torch

def main():
    print(f"当前环境: {settings.app_env}")
    print(f"计划使用显卡: {settings.cuda_visible_devices}")
    print(f"Torch 实际可用显卡数: {torch.cuda.device_count()}")
    
    if torch.cuda.is_available():
        print(f"当前使用的设备: {torch.cuda.get_device_name(0)}")

if __name__ == "__main__":
    main()