"""
可视化：
- 绘制训练过程中的 loss 和 accuracy 曲线
- 第一层权重矩阵可视化
- 错例分析
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns


class Visualizer:
    """
    可视化工具
    """
    def __init__(self, trainer, model, data_loader):
        """
        参数:
            trainer: Trainer 实例 (包含训练历史)
            model: ThreeLayerMLP 实例
            data_loader: FashionMNISTDataLoader 实例
        """
        self.trainer = trainer
        self.model = model
        self.data_loader = data_loader

    def plot_loss_curves(self, save_path=None):
        """
        绘制训练和验证 Loss 曲线
        """
        history = self.trainer.get_history()
        train_loss = history['train_loss']
        val_loss = history['val_loss']

        epochs = range(1, len(train_loss) + 1)
        val_epochs = range(1, len(val_loss) + 1) if val_loss else []

        plt.figure(figsize=(8, 5))
        plt.plot(epochs, train_loss, 'b-', label='Training Loss')
        if val_loss:
            plt.plot(val_epochs, val_loss, 'r-', label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)

        if save_path:
            plt.savefig(save_path, dpi=150)
        plt.show()

    def plot_accuracy_curve(self, save_path=None):
        """
        绘制验证集 Accuracy 曲线
        """
        history = self.trainer.get_history()
        val_acc = history['val_acc']

        if not val_acc:
            print("No validation accuracy data available.")
            return

        epochs = range(1, len(val_acc) + 1)

        plt.figure(figsize=(8, 5))
        plt.plot(epochs, val_acc, 'g-', marker='o', markersize=4, label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Validation Accuracy Curve')
        plt.ylim([0, 1])
        plt.legend()
        plt.grid(True)

        if save_path:
            plt.savefig(save_path, dpi=150)
        plt.show()

    def visualize_first_layer_weights(self, num_filters=36, save_path=None):
        """
        将第一层权重矩阵 (shape: 784, hidden_dim1) 的每一列恢复成 28x28 图像
        并展示前 num_filters 个 (或者随机选择)
        """
        # 获取第一层权重 (fc1.W)
        W1 = self.model.fc1.W  # shape (784, hidden_dim1)
        hidden_dim = W1.shape[1]

        # 选择要显示的滤波器数量
        if hidden_dim < num_filters:
            num_filters = hidden_dim
        # 均匀采样
        indices = np.linspace(0, hidden_dim-1, num_filters, dtype=int)

        # 归一化权重到 [0,1] 以便显示
        # 每个滤波器单独归一化
        filters = W1[:, indices].T  # (num_filters, 784)
        filters_norm = np.zeros_like(filters)
        for i in range(num_filters):
            f = filters[i]
            f_min, f_max = f.min(), f.max()
            if f_max - f_min > 1e-8:
                filters_norm[i] = (f - f_min) / (f_max - f_min)
            else:
                filters_norm[i] = f - f_min

        # 绘制网格
        cols = int(np.ceil(np.sqrt(num_filters)))
        rows = int(np.ceil(num_filters / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(cols*2, rows*2))
        axes = axes.flatten()

        for i in range(num_filters):
            img = filters_norm[i].reshape(28, 28)
            axes[i].imshow(img, cmap='gray')
            axes[i].axis('off')
            axes[i].set_title(f'Filter {indices[i]}')
        for i in range(num_filters, len(axes)):
            axes[i].axis('off')

        plt.suptitle(f'First Layer Weights Visualization (first {num_filters} filters)')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150)
        plt.show()

    def error_analysis(self, num_examples=10, save_path=None):
        """
        从测试集中找出分类错误的样本，展示图像、真实标签和预测标签
        """
        # 获取测试集所有样本和标签
        X_test, y_test = self.data_loader.get_test_data()
        y_pred = self.model.predict(X_test)

        # 找出错误样本的索引
        error_indices = np.where(y_pred != y_test)[0]
        print(f"Total errors: {len(error_indices)} / {len(y_test)} ({len(error_indices)/len(y_test)*100:.2f}%)")

        if len(error_indices) == 0:
            print("No errors found!")
            return

        # 选择前 num_examples 个错误样本 (或随机采样)
        if len(error_indices) > num_examples:
            selected = np.random.choice(error_indices, num_examples, replace=False)
        else:
            selected = error_indices

        # 类别标签文字
        class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                       'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

        # 绘制图像
        cols = 5
        rows = int(np.ceil(num_examples / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(cols*3, rows*3))
        axes = axes.flatten()

        for i, idx in enumerate(selected):
            img = X_test[idx].reshape(28, 28)
            true_label = y_test[idx]
            pred_label = y_pred[idx]
            axes[i].imshow(img, cmap='gray')
            axes[i].set_title(f'True: {class_names[true_label]}\nPred: {class_names[pred_label]}', fontsize=9)
            axes[i].axis('off')
        for i in range(len(selected), len(axes)):
            axes[i].axis('off')

        plt.suptitle(f'Error Analysis ({len(selected)} misclassified examples)')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150)
        plt.show()

        # 打印每个类别的错误率
        cm = confusion_matrix(y_test, y_pred, labels=range(10))
        class_errors = []
        for c in range(10):
            total_c = np.sum(y_test == c)
            if total_c > 0:
                err_c = cm[c, :].sum() - cm[c, c]
                class_errors.append(err_c / total_c)
            else:
                class_errors.append(0)
        print("\nError rate per class:")
        for c in range(10):
            print(f"  {class_names[c]}: {class_errors[c]*100:.2f}%")

    def plot_confusion_matrix(self, save_path=None):
        """
        绘制混淆矩阵 (需要先运行 trainer.test() 获取 cm)
        """
        X_test, y_test = self.data_loader.get_test_data()
        y_pred = self.model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred, labels=range(10))
        class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                       'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        if save_path:
            plt.savefig(save_path, dpi=150)
        plt.show()

    def visualize_first_layer_weights_improved(self, num_filters=None, 
                                            sort_by='none', 
                                            cmap='RdBu', 
                                            save_path=None):
        """
        改进的第一层权重可视化
        参数:
            num_filters: 要显示的滤波器数量，None 表示显示全部
            sort_by: 排序方式，'norm' 按 L2 范数，'std' 按标准差，'none' 不排序
            cmap: 颜色映射，建议 'RdBu' 或 'gray'
            save_path: 保存路径
        """
        W1 = self.model.fc1.W          # shape (784, hidden_dim1)
        hidden_dim = W1.shape[1]

        # 选择要显示的滤波器
        if num_filters is None:
            num_filters = hidden_dim
        else:
            num_filters = min(num_filters, hidden_dim)

        # 计算每个滤波器的统计量
        filters = W1.T                  # (hidden_dim, 784)
        if sort_by == 'norm':
            scores = np.linalg.norm(filters, axis=1)
            idx_sorted = np.argsort(scores)[::-1]   # 降序
        elif sort_by == 'std':
            scores = np.std(filters, axis=1)
            idx_sorted = np.argsort(scores)[::-1]
        else:
            idx_sorted = np.arange(hidden_dim)

        selected_idx = idx_sorted[:num_filters]

        # 归一化：全局归一化到 [-1, 1] 或各自归一化
        # 这里采用全局归一化，便于统一比较
        global_min = filters[selected_idx].min()
        global_max = filters[selected_idx].max()
        if global_max - global_min > 1e-8:
            filters_norm = (filters[selected_idx] - global_min) / (global_max - global_min) * 2 - 1
        else:
            filters_norm = filters[selected_idx] - global_min

        # 计算网格布局
        cols = int(np.ceil(np.sqrt(num_filters)))
        rows = int(np.ceil(num_filters / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(cols*2, rows*2))
        axes = axes.flatten()

        for i, idx in enumerate(selected_idx):
            img = filters_norm[i].reshape(28, 28)
            im = axes[i].imshow(img, cmap=cmap, interpolation='bilinear')
            axes[i].axis('off')
            axes[i].set_title(f'{idx}', fontsize=8)
        for i in range(num_filters, len(axes)):
            axes[i].axis('off')

        # 添加全局颜色条
        fig.subplots_adjust(right=0.9)
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        fig.colorbar(im, cax=cbar_ax)
        plt.suptitle(f'First Layer Weights (size={hidden_dim}, showing {num_filters} filters)\nGlobal normalize, cmap={cmap}')
        plt.tight_layout(rect=[0, 0, 0.9, 0.95])
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()



if __name__ == "__main__":
    from data_loader import FashionMNISTDataLoader
    from model import ThreeLayerMLP
    from trainer import Trainer

    DATA_DIR = r'C:\Users\86180\Downloads\CV-HW1\fashion-mnist-master\data\fashion'
    loader = FashionMNISTDataLoader(DATA_DIR, validation_split=0.1, random_seed=42)
    model = ThreeLayerMLP(hidden_dim1=128, hidden_dim2=64, hidden_activation='relu')
    trainer = Trainer(model, loader, learning_rate=0.01, weight_decay=1e-4,
                      lr_decay_strategy='step', lr_decay_step=10, verbose=False)
    trainer.train(num_epochs=5, batch_size=64, eval_interval=1)

    viz = Visualizer(trainer, model, loader)
    viz.plot_training_curves()
    viz.visualize_first_layer_weights(num_filters=36)
    # 测试准确率和混淆矩阵
    test_acc, cm = trainer.test()
    print(f"Test Acc: {test_acc:.4f}")
    viz.plot_confusion_matrix()
    viz.error_analysis(num_examples=10)