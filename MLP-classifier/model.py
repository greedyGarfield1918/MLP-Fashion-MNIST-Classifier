"""
模型定义：
- 激活函数 (ReLU, Sigmoid, Tanh)
- 全连接层 (线性变换 + 激活)
- 整个网络的前向、反向传播
- 支持 L2 正则化（在损失和梯度中体现）
"""

import numpy as np


class Activation:
    """激活函数及其导数"""
    @staticmethod
    def relu(x):
        return np.maximum(0, x)

    @staticmethod
    def relu_derivative(x):
        return (x > 0).astype(float)

    @staticmethod
    def sigmoid(x):
        # 防止溢出
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def sigmoid_derivative(x):
        sig = Activation.sigmoid(x)
        return sig * (1 - sig)

    @staticmethod
    def tanh(x):
        return np.tanh(x)

    @staticmethod
    def tanh_derivative(x):
        return 1 - np.tanh(x) ** 2


class FullyConnectedLayer:
    """
    全连接层: y = activation(W @ x + b)
    """
    def __init__(self, input_dim, output_dim, activation='relu'):
        """
        参数:
            input_dim: 输入维度
            output_dim: 输出维度
            activation: 激活函数名称 ('relu', 'sigmoid', 'tanh', 或 None 表示线性)
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation_name = activation

        # Xavier 初始化
        self.W = np.random.randn(input_dim, output_dim) * np.sqrt(2.0 / (input_dim + output_dim))
        self.b = np.zeros((1, output_dim))

        # 缓存前向传播的中间值 (用于反向传播)
        self.cache = None

        # 梯度 (用于优化器更新)
        self.dW = None
        self.db = None

        # 选择激活函数
        if activation == 'relu':
            self.activate = Activation.relu
            self.activate_deriv = Activation.relu_derivative
        elif activation == 'sigmoid':
            self.activate = Activation.sigmoid
            self.activate_deriv = Activation.sigmoid_derivative
        elif activation == 'tanh':
            self.activate = Activation.tanh
            self.activate_deriv = Activation.tanh_derivative
        elif activation is None:
            # 线性层 (无激活)
            self.activate = lambda x: x
            self.activate_deriv = lambda x: 1
        else:
            raise ValueError(f"Unsupported activation: {activation}")

    def forward(self, x, is_training=True):
        """
        前向传播
        参数:
            x: 输入 shape (batch_size, input_dim)
            is_training: 是否训练模式 (用于 dropout 等，目前未用)
        返回:
            out: 输出 shape (batch_size, output_dim)
        """
        z = x @ self.W + self.b          # 线性部分
        out = self.activate(z)           # 激活
        # 缓存 x, z 用于反向传播
        self.cache = (x, z)
        return out

    def backward(self, dout):
        """
        反向传播
        参数:
            dout: 上一层的梯度 (loss 对该层输出的导数) shape (batch_size, output_dim)
        返回:
            dx: 该层输入的梯度 (传递给前一层) shape (batch_size, input_dim)
        """
        x, z = self.cache
        # 激活函数的导数
        dact = dout * self.activate_deriv(z)   # shape (batch_size, output_dim)
        # 参数梯度
        self.dW = (x.T @ dact) / x.shape[0]    # 除以 batch_size 取平均
        self.db = np.mean(dact, axis=0, keepdims=True)
        # 输入梯度 (传给上一层)
        dx = dact @ self.W.T
        return dx

    def update_params(self, learning_rate, weight_decay=0.0):
        """
        使用 SGD 更新参数 (带 weight decay L2 正则化)
        参数:
            learning_rate: 学习率
            weight_decay: L2 正则化系数 (lambda)
        """
        # 梯度下降: W = W - lr * (dW + weight_decay * W)
        self.W -= learning_rate * (self.dW + weight_decay * self.W)
        self.b -= learning_rate * self.db


class ThreeLayerMLP:
    """
    三层 MLP: 输入层(784) -> 隐藏层1 -> 隐藏层2 -> 输出层(10)
    隐藏层大小可配置，激活函数可配置
    """
    def __init__(self, hidden_dim1=256, hidden_dim2=128, hidden_activation='relu', output_activation=None):
        """
        参数:
            hidden_dim1: 第一隐藏层神经元数
            hidden_dim2: 第二隐藏层神经元数
            hidden_activation: 隐藏层使用的激活函数 ('relu', 'sigmoid', 'tanh')
            output_activation: 输出层激活函数 (通常为 None，因为交叉熵损失配合 softmax 会单独处理)
        """
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2

        # 三层全连接层 (输出层无激活，因为后面会接 softmax + cross-entropy)
        self.fc1 = FullyConnectedLayer(784, hidden_dim1, activation=hidden_activation)
        self.fc2 = FullyConnectedLayer(hidden_dim1, hidden_dim2, activation=hidden_activation)
        self.fc3 = FullyConnectedLayer(hidden_dim2, 10, activation=output_activation)

        self.layers = [self.fc1, self.fc2, self.fc3]

    def forward(self, x, is_training=True):
        """
        前向传播
        参数:
            x: 输入 shape (batch_size, 784)
            is_training: 是否训练模式
        返回:
            logits: 输出层未经过 softmax 的分数 shape (batch_size, 10)
        """
        out = self.fc1.forward(x, is_training)
        out = self.fc2.forward(out, is_training)
        logits = self.fc3.forward(out, is_training)
        return logits

    def backward(self, dout):
        """
        反向传播
        参数:
            dout: loss 对 logits 的梯度 (即 softmax_cross_entropy 的梯度)
        """
        # 逐层反向传播
        dout = self.fc3.backward(dout)
        dout = self.fc2.backward(dout)
        dout = self.fc1.backward(dout)
        return dout

    def update_params(self, learning_rate, weight_decay=0.0):
        """更新所有层参数"""
        for layer in self.layers:
            layer.update_params(learning_rate, weight_decay)

    def get_params(self):
        """返回所有参数的副本 (用于调试或保存)"""
        params = []
        for layer in self.layers:
            params.append((layer.W.copy(), layer.b.copy()))
        return params

    def set_params(self, params):
        """设置参数 (用于加载最优模型)"""
        for layer, (W, b) in zip(self.layers, params):
            layer.W = W.copy()
            layer.b = b.copy()

    def compute_loss_and_grad(self, X, y_onehot, weight_decay=0.0):
        """
        给定一批数据，计算交叉熵损失和梯度
        参数:
            X: 输入 (batch_size, 784)
            y_onehot: one-hot 标签 (batch_size, 10)
            weight_decay: L2 正则化系数
        返回:
            loss: 标量损失
            grad_input: 对输入的梯度 (本任务不需要，但可用于检查)
        """
        batch_size = X.shape[0]
        logits = self.forward(X)

        # Softmax 稳定计算
        logits_shifted = logits - np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(logits_shifted)
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        # 交叉熵损失
        cross_entropy = -np.mean(np.sum(y_onehot * np.log(probs + 1e-15), axis=1))

        # L2 正则化损失
        l2_loss = 0.0
        for layer in self.layers:
            l2_loss += 0.5 * weight_decay * np.sum(layer.W ** 2)
        total_loss = cross_entropy + l2_loss

        # 计算梯度: dLoss/dLogits = probs - y_onehot
        dlogits = (probs - y_onehot) / batch_size   # 除以 batch_size 是为了与损失中平均对应
        # 反向传播
        self.backward(dlogits)

        return total_loss

    def predict(self, X):
        """
        预测类别
        参数:
            X: 输入 (n_samples, 784)
        返回:
            pred: 预测的类别索引 (n_samples,)
        """
        logits = self.forward(X, is_training=False)
        return np.argmax(logits, axis=1)

    def accuracy(self, X, y):
        """计算分类准确率"""
        pred = self.predict(X)
        return np.mean(pred == y)


if __name__ == "__main__":
    np.random.seed(42)
    model = ThreeLayerMLP(hidden_dim1=128, hidden_dim2=64, hidden_activation='relu')

    X_batch = np.random.randn(32, 784).astype(np.float32)
    y_batch = np.random.randint(0, 10, size=(32,))
    y_onehot = np.eye(10)[y_batch]

    loss = model.compute_loss_and_grad(X_batch, y_onehot, weight_decay=1e-4)
    print(f"Initial loss: {loss:.6f}")

    model.update_params(learning_rate=0.01, weight_decay=1e-4)

    loss2 = model.compute_loss_and_grad(X_batch, y_onehot, weight_decay=1e-4)
    print(f"Loss after one update: {loss2:.6f}")

    acc = model.accuracy(X_batch, y_batch)
    print(f"Random accuracy: {acc:.4f} (expected ~0.1)")