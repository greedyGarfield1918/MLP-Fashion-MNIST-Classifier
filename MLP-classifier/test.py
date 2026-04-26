import os
import pickle
from data_loader import FashionMNISTDataLoader
from model import ThreeLayerMLP

def main():
    base_dir = os.path.dirname(__file__)
    DATA_DIR = os.path.join(base_dir, '..', 'fashion-mnist-master', 'data', 'fashion')
    loader = FashionMNISTDataLoader(DATA_DIR)
    model = ThreeLayerMLP(hidden_dim1=256, hidden_dim2=128)
    with open('./saved_models/best_model.pkl', 'rb') as f:
        params = pickle.load(f)
        model.set_params(params)
    acc, cm = model.accuracy(loader.get_test_data()[0], loader.get_test_data()[1])
    print(f"Test Acc: {acc:.4f}")
    print("Confusion Matrix:\n", cm)

if __name__ == '__main__':
    main()