"""
数据加载与预处理：
- 从本地 Fashion-MNIST 文件读取图像和标签
- 归一化、打乱数据
- 划分训练集、验证集、测试集
- 提供批量迭代器
"""

import os
import gzip
import numpy as np
from sklearn.model_selection import train_test_split


class FashionMNISTDataLoader:
    """
    Fashion-MNIST 数据加载器
    """

    def __init__(self, data_dir, validation_split=0.1, random_seed=42):
        """
        参数:
            data_dir (str): 包含四个 .gz 文件的目录路径
            validation_split (float): 从训练集中划分验证集的比例，默认 0.1
            random_seed (int): 随机种子，确保可复现性
        """
        self.data_dir = data_dir
        self.validation_split = validation_split
        self.random_seed = random_seed

        # 加载原始数据 (图像展平为 784 维，标签为整数 0~9)
        self.X_train_raw, self.y_train_raw = self._load_data(kind='train')
        self.X_test_raw, self.y_test_raw = self._load_data(kind='t10k')

        # 归一化 (像素值 0~255 -> 0~1)
        self.X_train_raw = self.X_train_raw.astype(np.float32) / 255.0
        self.X_test_raw = self.X_test_raw.astype(np.float32) / 255.0

        # 划分训练集和验证集
        self._split_train_val()

    def _load_data(self, kind):
        """
        内部方法：从 .gz 文件加载图像和标签
        参数:
            kind (str): 'train' 或 't10k'
        返回:
            images (np.ndarray): shape (n_samples, 784), dtype=uint8
            labels (np.ndarray): shape (n_samples,), dtype=uint8
        """
        labels_path = os.path.join(self.data_dir, f'{kind}-labels-idx1-ubyte.gz')
        images_path = os.path.join(self.data_dir, f'{kind}-images-idx3-ubyte.gz')

        with gzip.open(labels_path, 'rb') as lbpath:
            labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)

        with gzip.open(images_path, 'rb') as imgpath:
            images = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16)
            images = images.reshape(len(labels), 784)

        return images, labels

    def _split_train_val(self):
        """
        从训练集中划分出验证集 (分层采样，保持类别比例)
        """
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            self.X_train_raw, self.y_train_raw,
            test_size=self.validation_split,
            stratify=self.y_train_raw,          # 按标签分层
            random_state=self.random_seed
        )

    def get_train_data(self):
        """返回训练集 (图像, 标签)"""
        return self.X_train, self.y_train

    def get_val_data(self):
        """返回验证集 (图像, 标签)"""
        return self.X_val, self.y_val

    def get_test_data(self):
        """返回测试集 (图像, 标签)"""
        return self.X_test_raw, self.y_test_raw

    def create_batch_generator(self, X, y, batch_size, shuffle=True):
        """
        生成批量数据 (生成器函数)
        参数:
            X (np.ndarray): 图像数组
            y (np.ndarray): 标签数组
            batch_size (int): 批量大小
            shuffle (bool): 是否在每个 epoch 前打乱数据
        返回:
            生成器，每次 yield (batch_X, batch_y)
        """
        n_samples = X.shape[0]
        indices = np.arange(n_samples)

        if shuffle:
            np.random.shuffle(indices)

        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            batch_indices = indices[start_idx:end_idx]
            yield X[batch_indices], y[batch_indices]

    def get_train_batches(self, batch_size, shuffle=True):
        """训练集批量生成器"""
        return self.create_batch_generator(self.X_train, self.y_train, batch_size, shuffle)

    def get_val_batches(self, batch_size, shuffle=False):
        """验证集批量生成器 (通常不需要打乱)"""
        return self.create_batch_generator(self.X_val, self.y_val, batch_size, shuffle)

    def get_test_batches(self, batch_size, shuffle=False):
        """测试集批量生成器"""
        return self.create_batch_generator(self.X_test_raw, self.y_test_raw, batch_size, shuffle)


if __name__ == "__main__":
    DATA_DIR = r'C:\Users\86180\Downloads\CV-HW1\fashion-mnist-master\data\fashion'

    loader = FashionMNISTDataLoader(DATA_DIR, validation_split=0.1, random_seed=42)

    X_tr, y_tr = loader.get_train_data()
    X_val, y_val = loader.get_val_data()
    X_te, y_te = loader.get_test_data()

    print(f"训练集大小: {X_tr.shape}, 标签: {y_tr.shape}")
    print(f"验证集大小: {X_val.shape}, 标签: {y_val.shape}")
    print(f"测试集大小: {X_te.shape}, 标签: {y_te.shape}")

    # 测试批量生成器
    batch_gen = loader.get_train_batches(batch_size=64, shuffle=True)
    for batch_X, batch_y in batch_gen:
        print(f"Batch shape: {batch_X.shape}, labels: {batch_y.shape}")
        break   # 只打印第一组