"""
训练过程：
- 执行训练循环
- 学习率衰减 (step decay / exponential decay)
- 验证集评估
- 保存最优模型权重
- 记录训练历史 (loss, accuracy)
"""

import numpy as np
import copy
import time


class Trainer:
    """
    三层 MLP 训练器
    """
    def __init__(self, model, data_loader, learning_rate=0.01, weight_decay=0.0,
                 lr_decay_strategy='step', lr_decay_step=20, lr_decay_rate=0.9,
                 verbose=True):
        """
        参数:
            model: ThreeLayerMLP 实例
            data_loader: FashionMNISTDataLoader 实例
            learning_rate: 初始学习率
            weight_decay: L2 正则化系数
            lr_decay_strategy: 学习率衰减策略 ('step' 或 'exponential' 或 None)
            lr_decay_step: step 衰减的 epoch 间隔
            lr_decay_rate: 衰减因子 (每次乘以该值)
            verbose: 是否打印训练过程
        """
        self.model = model
        self.data_loader = data_loader
        self.base_lr = learning_rate
        self.current_lr = learning_rate
        self.weight_decay = weight_decay
        self.lr_decay_strategy = lr_decay_strategy
        self.lr_decay_step = lr_decay_step
        self.lr_decay_rate = lr_decay_rate
        self.verbose = verbose

        # 历史记录
        self.train_losses = []
        self.val_losses = []
        self.val_accs = []

        # 最佳模型
        self.best_val_acc = 0.0
        self.best_model_params = None
        self.best_epoch = -1

    def _update_learning_rate(self, epoch):
        """根据策略更新当前学习率"""
        if self.lr_decay_strategy == 'step':
            # 每隔 lr_decay_step 个 epoch 衰减一次
            steps = epoch // self.lr_decay_step
            self.current_lr = self.base_lr * (self.lr_decay_rate ** steps)
        elif self.lr_decay_strategy == 'exponential':
            # 指数衰减: lr = base_lr * exp(-decay_rate * epoch)
            self.current_lr = self.base_lr * np.exp(-self.lr_decay_rate * epoch)
        else:  # None 或其他
            self.current_lr = self.base_lr

    def train_epoch(self, batch_size=64):
        """
        训练一个 epoch (遍历整个训练集一次)
        返回: 平均训练损失
        """
        epoch_loss = 0.0
        num_batches = 0

        for batch_X, batch_y in self.data_loader.get_train_batches(batch_size, shuffle=True):
            # one-hot 编码
            y_onehot = np.eye(10)[batch_y]

            # 计算损失和梯度 (内部会更新梯度缓存)
            loss = self.model.compute_loss_and_grad(batch_X, y_onehot, self.weight_decay)
            # 更新参数
            self.model.update_params(self.current_lr, self.weight_decay)

            epoch_loss += loss
            num_batches += 1

        avg_loss = epoch_loss / num_batches
        return avg_loss

    def validate(self, batch_size=64):
        """
        在验证集上计算平均损失和准确率
        返回: (avg_loss, accuracy)
        """
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        num_batches = 0

        for batch_X, batch_y in self.data_loader.get_val_batches(batch_size, shuffle=False):
            y_onehot = np.eye(10)[batch_y]
            # 验证时不需要计算梯度，也不需要更新，但需要损失值
            logits = self.model.forward(batch_X, is_training=False)
            # 手动计算交叉熵损失
            probs = self._softmax(logits)
            cross_entropy = -np.mean(np.sum(y_onehot * np.log(probs + 1e-15), axis=1))
            val_loss += cross_entropy
            num_batches += 1

            # 准确率
            pred = np.argmax(logits, axis=1)
            val_correct += np.sum(pred == batch_y)
            val_total += batch_y.shape[0]

        avg_loss = val_loss / num_batches
        accuracy = val_correct / val_total
        return avg_loss, accuracy

    def _softmax(self, logits):
        """稳定 softmax"""
        logits_shifted = logits - np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(logits_shifted)
        return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    def save_best_model(self):
        """保存当前模型参数为最优"""
        self.best_model_params = self.model.get_params()

    def restore_best_model(self):
        """恢复最优模型参数"""
        if self.best_model_params is not None:
            self.model.set_params(self.best_model_params)
            if self.verbose:
                print(f"Restored best model from epoch {self.best_epoch} with val acc {self.best_val_acc:.4f}")

    def train(self, num_epochs, batch_size=64, eval_interval=1, save_best=True):
        """
        完整训练过程
        参数:
            num_epochs: 训练轮数
            batch_size: 批量大小
            eval_interval: 每隔多少个 epoch 在验证集上评估一次
            save_best: 是否保存验证集上准确率最高的模型
        """
        if self.verbose:
            print(f"Start training for {num_epochs} epochs, batch_size={batch_size}")
            print(f"Initial learning rate: {self.base_lr}, decay: {self.lr_decay_strategy}")
            print("-" * 60)

        for epoch in range(1, num_epochs + 1):
            # 更新学习率
            self._update_learning_rate(epoch)

            # 训练一个 epoch
            start_time = time.time()
            train_loss = self.train_epoch(batch_size)
            epoch_time = time.time() - start_time

            # 记录训练损失
            self.train_losses.append(train_loss)

            # 定期评估验证集
            if epoch % eval_interval == 0:
                val_loss, val_acc = self.validate(batch_size)
                self.val_losses.append(val_loss)
                self.val_accs.append(val_acc)

                # 保存最佳模型
                if save_best and val_acc > self.best_val_acc:
                    self.best_val_acc = val_acc
                    self.best_epoch = epoch
                    self.save_best_model()

                if self.verbose:
                    print(f"Epoch {epoch:3d}/{num_epochs} | LR: {self.current_lr:.5f} | "
                          f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                          f"Val Acc: {val_acc:.4f} | Time: {epoch_time:.2f}s")
            else:
                if self.verbose:
                    print(f"Epoch {epoch:3d}/{num_epochs} | LR: {self.current_lr:.5f} | "
                          f"Train Loss: {train_loss:.4f} | Time: {epoch_time:.2f}s")

        if self.verbose:
            print("-" * 60)
            print(f"Training finished. Best val acc: {self.best_val_acc:.4f} at epoch {self.best_epoch}")

        # 恢复最佳模型用于后续测试
        if save_best and self.best_model_params is not None:
            self.restore_best_model()

    def test(self, batch_size=64):
        """
        在测试集上评估当前模型 (假设已经恢复最优模型)
        返回: (accuracy, confusion_matrix)
        """
        y_true = []
        y_pred = []

        for batch_X, batch_y in self.data_loader.get_test_batches(batch_size, shuffle=False):
            logits = self.model.forward(batch_X, is_training=False)
            pred = np.argmax(logits, axis=1)
            y_true.extend(batch_y)
            y_pred.extend(pred)

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        accuracy = np.mean(y_true == y_pred)

        # 计算混淆矩阵 (使用 sklearn 的 confusion_matrix)
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_true, y_pred, labels=range(10))

        return accuracy, cm

    def get_history(self):
        """返回训练历史字典"""
        return {
            'train_loss': self.train_losses,
            'val_loss': self.val_losses,
            'val_acc': self.val_accs
        }


if __name__ == "__main__":
    from data_loader import FashionMNISTDataLoader
    from model import ThreeLayerMLP

    DATA_DIR = r'C:\Users\86180\Downloads\CV-HW1\fashion-mnist-master\data\fashion'

    loader = FashionMNISTDataLoader(DATA_DIR, validation_split=0.1, random_seed=42)

    model = ThreeLayerMLP(hidden_dim1=256, hidden_dim2=128, hidden_activation='relu')

    trainer = Trainer(model, loader, learning_rate=0.1, weight_decay=1e-4,
                      lr_decay_strategy='step', lr_decay_step=10, lr_decay_rate=0.5,
                      verbose=True)

    trainer.train(num_epochs=5, batch_size=64, eval_interval=1, save_best=True)

    test_acc, cm = trainer.test()
    print(f"\nTest Accuracy: {test_acc:.4f}")
    print("Confusion Matrix:")
    print(cm)