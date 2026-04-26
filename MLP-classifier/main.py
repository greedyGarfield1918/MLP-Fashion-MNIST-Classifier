import os
import pickle
from data_loader import FashionMNISTDataLoader
from model import ThreeLayerMLP
from trainer import Trainer
from search import HyperparameterSearch
from visualize import Visualizer


def main():
    base_dir = os.path.dirname(__file__)
    DATA_DIR = os.path.join(base_dir, '..', 'fashion-mnist-master', 'data', 'fashion')
    SAVE_DIR = './saved_models'
    os.makedirs(SAVE_DIR, exist_ok=True)

    # 1. 加载数据
    print("Loading data...")
    loader = FashionMNISTDataLoader(DATA_DIR, validation_split=0.1, random_seed=42)

    # 2. 超参数搜索
    print("\n=== Hyperparameter Search ===")
    searcher = HyperparameterSearch(loader, search_type='grid', verbose=False)
    param_grid = {
        'learning_rate': [0.01, 0.05],
        'hidden_dim1': [128, 256],
        'hidden_dim2': [64, 128],
        'weight_decay': [0, 1e-4]
    }
    best_params, _ = searcher.grid_search(param_grid, num_epochs=5, batch_size=128)
    print(f"Best hyperparameters: {best_params}")
    
    # 3. 使用最佳超参数训练最终模型
    print("\n=== Training Final Model ===")
    model = ThreeLayerMLP(
        hidden_dim1=best_params['hidden_dim1'],
        hidden_dim2=best_params['hidden_dim2'],
        hidden_activation='relu'
    )
    trainer = Trainer(
        model, loader,
        learning_rate=best_params['learning_rate'],
        weight_decay=best_params['weight_decay'],
        lr_decay_strategy='step',
        lr_decay_step=15,
        lr_decay_rate=0.8,
        verbose=True
    )
    trainer.train(num_epochs=50, batch_size=64, eval_interval=1, save_best=True)

    # 保存模型权重
    model_path = os.path.join(SAVE_DIR, 'best_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model.get_params(), f)
    print(f"Model saved to {model_path}")

    # 4. 测试
    print("\n=== Testing ===")
    test_acc, cm = trainer.test()
    print(f"Test Accuracy: {test_acc:.4f}")

    # 5. 可视化
    viz = Visualizer(trainer, model, loader)
    viz.plot_loss_curves(save_path='loss_curves.png')
    viz.plot_accuracy_curve(save_path='accuracy_curve.png')
    viz.visualize_first_layer_weights(num_filters=36, save_path='first_layer_weights.png')
    viz.plot_confusion_matrix(save_path='confusion_matrix.png')
    viz.error_analysis(num_examples=12, save_path='error_analysis.png')
    viz.visualize_first_layer_weights_improved(
        num_filters=64, 
        sort_by='norm', 
        cmap='RdBu', 
        save_path='first_layer_weights_improved.png'
    )


if __name__ == "__main__":
    main()