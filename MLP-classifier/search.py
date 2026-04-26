"""
超参数搜索 (网格搜索 / 随机搜索)
- 每个组合训练时保存最佳模型
- 使用 trainer.best_val_acc 作为该组合的性能指标
- 支持固定随机种子
"""

import itertools
import numpy as np
from trainer import Trainer
from model import ThreeLayerMLP


class HyperparameterSearch:
    def __init__(self, data_loader, search_type='grid', verbose=False, seed=42):
        self.data_loader = data_loader
        self.search_type = search_type
        self.verbose = verbose
        self.base_seed = seed
        self.results = []

    def grid_search(self, param_grid, num_epochs=5, batch_size=64, eval_interval=1,
                    lr_decay_strategy=None, lr_decay_step=10, lr_decay_rate=0.8):
        """
        网格搜索
        """
        keys = param_grid.keys()
        values = param_grid.values()
        combos = list(itertools.product(*values))
        total = len(combos)

        print(f"Grid search: {total} combinations to try (each for {num_epochs} epochs)")
        print("-" * 60)

        for idx, combo in enumerate(combos):
            params = dict(zip(keys, combo))

            # 为每个组合设置不同的随机种子（保证可复现）
            np.random.seed(self.base_seed + idx)

            if self.verbose:
                print(f"\n[{idx+1}/{total}] Testing params: {params}")

            model = ThreeLayerMLP(
                hidden_dim1=params.get('hidden_dim1', 256),
                hidden_dim2=params.get('hidden_dim2', 128),
                hidden_activation='relu'
            )
            trainer = Trainer(
                model, self.data_loader,
                learning_rate=params['learning_rate'],
                weight_decay=params.get('weight_decay', 0.0),
                lr_decay_strategy=lr_decay_strategy,   # 搜索时可关闭衰减
                lr_decay_step=lr_decay_step,
                lr_decay_rate=lr_decay_rate,
                verbose=False
            )
            # 训练并保存最佳模型
            trainer.train(num_epochs=num_epochs, batch_size=batch_size,
                          eval_interval=eval_interval, save_best=True)

            # 获取最佳验证准确率（trainer 中已记录）
            best_val_acc = trainer.best_val_acc if hasattr(trainer, 'best_val_acc') else 0.0
            self.results.append((params, best_val_acc))
            print(f"  -> Best Val Acc: {best_val_acc:.4f}")

        # 找出最佳组合
        best_idx = np.argmax([acc for _, acc in self.results])
        best_params, best_acc = self.results[best_idx]
        print("\n" + "="*60)
        print(f"Best params: {best_params}")
        print(f"Best validation accuracy: {best_acc:.4f}")
        return best_params, self.results

    def random_search(self, param_distributions, n_iter=10, num_epochs=5, batch_size=64,
                      lr_decay_strategy=None, lr_decay_step=10, lr_decay_rate=0.8,
                      random_seed=42):
        """
        随机搜索
        """
        np.random.seed(random_seed)
        print(f"Random search: {n_iter} combinations to try (each for {num_epochs} epochs)")
        print("-" * 60)

        for i in range(n_iter):
            params = {}
            for key, values in param_distributions.items():
                if isinstance(values, (list, range)):
                    idx = np.random.randint(0, len(values))
                    params[key] = values[idx]
                else:
                    raise ValueError(f"Unsupported distribution type for {key}")

            # 固定随机种子（可选，也可以不固定）
            np.random.seed(random_seed + i)

            if self.verbose:
                print(f"\n[{i+1}/{n_iter}] Testing params: {params}")

            model = ThreeLayerMLP(
                hidden_dim1=params.get('hidden_dim1', 256),
                hidden_dim2=params.get('hidden_dim2', 128),
                hidden_activation='relu'
            )
            trainer = Trainer(
                model, self.data_loader,
                learning_rate=params['learning_rate'],
                weight_decay=params.get('weight_decay', 0.0),
                lr_decay_strategy=lr_decay_strategy,
                lr_decay_step=lr_decay_step,
                lr_decay_rate=lr_decay_rate,
                verbose=False
            )
            trainer.train(num_epochs=num_epochs, batch_size=batch_size,
                          eval_interval=1, save_best=True)

            best_val_acc = trainer.best_val_acc if hasattr(trainer, 'best_val_acc') else 0.0
            self.results.append((params, best_val_acc))
            print(f"  -> Best Val Acc: {best_val_acc:.4f}")

        best_idx = np.argmax([acc for _, acc in self.results])
        best_params, best_acc = self.results[best_idx]
        print("\n" + "="*60)
        print(f"Best params: {best_params}")
        print(f"Best validation accuracy: {best_acc:.4f}")
        return best_params, self.results


# 测试代码
if __name__ == "__main__":
    from data_loader import FashionMNISTDataLoader

    DATA_DIR = r'C:\Users\86180\Downloads\CV-HW1\fashion-mnist-master\data\fashion'
    loader = FashionMNISTDataLoader(DATA_DIR, validation_split=0.1, random_seed=42)

    searcher = HyperparameterSearch(loader, search_type='grid', verbose=True, seed=42)

    param_grid = {
        'learning_rate': [0.01, 0.05, 0.1],
        'hidden_dim1': [128, 256],
        'hidden_dim2': [64, 128],
        'weight_decay': [0, 1e-4]
    }
    # 增加 num_epochs 到 5 或 10，让模型充分训练
    best_params, results = searcher.grid_search(param_grid, num_epochs=5, batch_size=128,
                                                lr_decay_strategy=None)  # 搜索时关闭衰减