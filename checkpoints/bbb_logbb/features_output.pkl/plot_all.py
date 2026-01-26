import pickle
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import pandas as pd

# 假设文件已上传到当前目录
FILE_PATHS = {
    'train': "bbb_logbb_train_features.pkl",
    'valid': "bbb_logbb_valid_features.pkl",
    'test': "bbb_logbb_test_features.pkl",
}

# 1. 加载数据
all_features = []
all_labels = []
all_splits = []

for split_name, path in FILE_PATHS.items():
    try:
        with open(path, 'rb') as f:
            data = pickle.load(f)
            features = data['features']
            labels = np.squeeze(data['y_true'])

            all_features.append(features)
            all_labels.append(labels)
            all_splits.extend([split_name] * len(labels))

    except FileNotFoundError:
        print(f"[错误] 文件未找到: {path}. 请上传该文件。")
        # 无法加载文件，退出绘图逻辑
        all_features = None
        break

if all_features is not None:
    # 2. 合并数据
    X_combined = np.concatenate(all_features, axis=0)
    y_combined = np.concatenate(all_labels, axis=0)

    # 3. 执行 t-SNE
    tsne = TSNE(n_components=2, perplexity=30, n_iter=3000, random_state=42, learning_rate='auto', init='pca')
    X_2d = tsne.fit_transform(X_combined)

    # 4. 准备绘图数据
    plot_df = pd.DataFrame({
        'TSNE-1': X_2d[:, 0],
        'TSNE-2': X_2d[:, 1],
        'Label': y_combined.astype(int),
        'Split': all_splits
    })

    # 5. 绘图
    plt.figure(figsize=(12, 10))
    sns.scatterplot(
        x='TSNE-1',
        y='TSNE-2',
        hue='Label',
        style='Split',
        data=plot_df,
        palette={0: '#3498db', 1: '#e74c3c'},
        s=50,
        alpha=0.7
    )

    plt.title('t-SNE Visualization of Combined ArcMol Features', fontsize=16)
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.grid(True, alpha=0.3)
    plt.legend(title='Category', bbox_to_anchor=(1.05, 1), loc='upper left')

    output_filename = 'combined_features_tsne.png'
    plt.savefig(output_filename, bbox_inches='tight')
    print(f"\n成功生成 t-SNE 图像: {output_filename}")