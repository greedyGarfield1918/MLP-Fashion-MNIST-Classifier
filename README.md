# MLP Fashion-MNIST Classifier

手工实现的三层神经网络（MLP）分类器，用于 Fashion-MNIST 服装图像分类。**仅使用 NumPy**，不依赖 PyTorch/TensorFlow 等自动微分框架。包含完整的反向传播、SGD 优化器、学习率衰减、L2 正则化、超参数搜索、可视化等功能。

## 环境依赖

- Python 3.8+
- NumPy
- Matplotlib
- scikit-learn

安装依赖：
```bash
pip install -r requirements.txt
```

## 数据集准备

本仓库不包含数据集文件。请从 [Fashion-MNIST 官方仓库](https://github.com/zalandoresearch/fashion-mnist) 下载以下四个文件，并将其放置在 **项目根目录的上两级目录** `fashion-mnist-master/data/fashion/` 中（相对于 `MLP-classifier` 文件夹）：

- `train-images-idx3-ubyte.gz`
- `train-labels-idx1-ubyte.gz`
- `t10k-images-idx3-ubyte.gz`
- `t10k-labels-idx1-ubyte.gz`

推荐的目录结构：
```
CV-HW1/                          # 根工作目录
├── fashion-mnist-master/
│   └── data/
│       └── fashion/             # 四个 .gz 文件
└── MLP-classifier/              # 代码仓库
    ├── main.py
    ├── test.py
    ├── ...
...
```

> 如果数据集放在其他位置，请修改 `main.py` 和 `test.py` 中的 `DATA_DIR` 变量。

## 运行说明

### 1. 完整训练 + 超参数搜索 + 测试 + 可视化

```bash
python main.py
```

该脚本会：
- 自动进行网格搜索（16 种组合，每个组合训练 5 个 epoch）选择最佳超参数。
- 使用最佳超参数训练最终模型（默认 50 个 epoch，SGD + Step Decay）。
- 保存最优模型权重到 `./saved_models/best_model.pkl`。
- 在测试集上输出准确率和混淆矩阵。
- 生成以下可视化图片：
  - 训练/验证损失曲线 (`loss_curves.png`)
  - 验证准确率曲线 (`accuracy_curve.png`)
  - 第一层权重图（36 个滤波器 + 改进版）(`first_layer_weights.png`, `first_layer_weights_improved.png`)
  - 混淆矩阵热力图 (`confusion_matrix.png`)
  - 错例分析（12 张）(`error_analysis.png`)

### 2. 仅测试已训练好的模型

如果已有保存的模型权重（位于 `./saved_models/best_model.pkl`），可以直接运行：

```bash
python test.py
```

该脚本会加载模型并在测试集上输出准确率和混淆矩阵。

### 3. 自定义超参数搜索

修改 `main.py` 中的 `param_grid` 字典，可调整搜索空间。例如：
```python
param_grid = {
    'learning_rate': [0.01, 0.05, 0.1],
    'hidden_dim1': [128, 256, 512],
    'hidden_dim2': [64, 128, 256],
    'weight_decay': [0, 1e-4, 1e-3]
}
```

### 4. 调整训练轮数或学习率衰减

在 `main.py` 中修改 `trainer.train()` 的参数。例如：
```python
trainer.train(num_epochs=100, batch_size=64, eval_interval=1, save_best=True)
```

学习率衰减策略可在 `Trainer` 初始化时设置：
- `lr_decay_strategy='step'`（默认，每 `lr_decay_step` 个 epoch 乘以 `lr_decay_rate`）
- `lr_decay_strategy='exponential'`
- `lr_decay_strategy=None`（无衰减）

## 预训练模型权重

训练好的最佳模型权重可从以下链接下载（请替换为实际 Google Drive 链接）：

[下载 best_model.pkl](https://drive.google.com/file/d/1UPkNlrTV4gdaFHUfmYDce4EvcJVbAJcR/view?usp=sharing)

把下载的 `best_model.pkl` 放入 `./saved_models/` 目录后，即可使用 `test.py` 进行测试。

## 实验结果摘要

- **最佳验证准确率**：84.10%（epoch 48）
- **测试准确率**：82.41%
- **各类别错误率**（最高为 Shirt：52.4%）

详细曲线、权重可视化和错例分析请参见实验报告。

## 代码结构

| 文件 | 说明 |
|------|------|
| `data_loader.py` | 加载 Fashion-MNIST，归一化，划分训练/验证/测试集，批量生成器 |
| `model.py` | 激活函数、全连接层、三层 MLP，手动实现前向/反向传播、L2 正则化 |
| `trainer.py` | 训练循环、SGD、学习率衰减、验证集评估、保存最佳模型 |
| `search.py` | 网格搜索 / 随机搜索超参数 |
| `visualize.py` | 训练曲线、权重可视化、混淆矩阵、错例分析 |
| `main.py` | 完整流程（搜索 → 训练 → 测试 → 可视化） |
| `test.py` | 单独加载模型并测试 |
| `requirements.txt` | 依赖列表 |

## 作业要求满足情况

- 手工实现自动微分与反向传播（仅使用 NumPy）
- 模块化设计（数据、模型、训练、搜索、可视化、主流程）
- 支持自定义隐藏层大小、ReLU/Sigmoid/Tanh 激活函数切换
- SGD 优化器、学习率衰减、交叉熵损失 + L2 正则化
- 根据验证集准确率自动保存最佳模型
- 网格搜索 / 随机搜索调优超参数
- 测试集准确率 + 混淆矩阵输出
- 可视化：Loss / Accuracy 曲线、第一层权重图像、错例分析

## 参考资料

- [Fashion-MNIST 数据集](https://github.com/zalandoresearch/fashion-mnist)
- [Xavier 初始化](http://proceedings.mlr.press/v9/glorot10a.html)

## 许可证

MIT
