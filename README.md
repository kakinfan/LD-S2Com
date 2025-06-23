# LD-S2Com Baseline

## 运行方法

1. 准备MFNet数据集，目录结构如下：
   - MFNet/
     - RGB/
     - LABEL/

2. 安装依赖
    pip install -r requirements.txt


3. 训练编码器与判别器
    python train_encoder.py

--train_encoder.py 是最小可运行训练主脚本，后续可扩展更多功能。

--数据集路径和label处理方式请根据实际情况调整。