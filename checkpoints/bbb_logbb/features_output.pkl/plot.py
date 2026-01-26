import pickle
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import pandas as pd

# 设置科研绘图风格
sns.set_theme(style="whitegrid", font_scale=1.2)

# 假设文件已上传到当前目录
FILE_PATHS = {
    'train': "bbb_logbb_train_features.pkl",
    'valid': "bbb_logbb_valid_features.pkl",
    'test': "bbb_logbb_test_features.pkl",
}
SPLITS_TO_PLOT = ['train', 'valid', 'test']

# 1. 加载所有数据
all_features = []
all_labels = []
all_splits = []
data_loaded = True

print("--- Step 1: Loading all data splits ---")
for split_name, path in FILE_PATHS.items():
    try:
        with open(path, 'rb') as f:
            data = pickle.load(f)
            features = data['features']
            labels = np.squeeze(data['y_true'])

            all_features.append(features)
            all_labels.append(labels)
            all_splits.extend([split_name] * len(labels))
            print(f"Loaded {split_name.upper()}: {features.shape[0]} samples.")

    except FileNotFoundError:
        print(f"[ERROR] 文件未找到: {path}. 请上传该文件。")
        data_loaded = False
        break

if not data_loaded:
    print("\nAborting plotting due to missing files.")
else:
    # 2. 合并数据
    X_combined = np.concatenate(all_features, axis=0)
    y_combined = np.concatenate(all_labels, axis=0)
    print(f"\nTotal combined samples: {X_combined.shape[0]}")

    # 3. 执行 t-SNE (在所有数据上降维，保证投影一致性)
    print("--- Step 2: Running t-SNE on combined features ---")
    tsne = TSNE(n_components=2, perplexity=30, n_iter=3000, random_state=42, learning_rate='auto', init='pca',
                n_jobs=-1)
    X_2d = tsne.fit_transform(X_combined)

    # 4. 准备绘图数据框
    plot_df = pd.DataFrame({
        'TSNE-1': X_2d[:, 0],
        'TSNE-2': X_2d[:, 1],
        'Label': y_combined.astype(int).astype(str),  # 转换为字符串用于绘图
        'Split': all_splits
    })

    # 5. 循环绘制并保存三个独立的图
    print("\n--- Step 3: Generating individual plots ---")

    # 确定全局的最大和最小坐标，用于统一图的边界 (可选，但更科学)
    x_min, x_max = X_2d[:, 0].min(), X_2d[:, 0].max()
    y_min, y_max = X_2d[:, 1].min(), X_2d[:, 1].max()

    for split in SPLITS_TO_PLOT:
        df_split = plot_df[plot_df['Split'] == split].copy()

        plt.figure(figsize=(8, 7))  # 统一图的大小

        # 绘图：仅显示当前 split 的点，按 Label 着色
        sns.scatterplot(
            x='TSNE-1',
            y='TSNE-2',
            hue='Label',
            data=df_split,
            palette={'0': '#3498db', '1': '#e74c3c'},  # 蓝色(0) 和 红色(1)
            s=40,
            alpha=0.7,
            linewidth=0.5,
            edgecolor='w'
        )

        # 设置坐标轴范围，确保所有图的尺度一致
        plt.xlim(x_min - 1, x_max + 1)
        plt.ylim(y_min - 1, y_max + 1)

        plt.title(f't-SNE Projection: {split.upper()} Set', fontsize=16)
        plt.xlabel('t-SNE Component 1', fontsize=14)
        plt.ylabel('t-SNE Component 2', fontsize=14)

        # 调整图例
        plt.legend(title='Class Label', loc='best', frameon=True)

        # 移除顶部和右侧边框
        sns.despine(trim=True)

        output_filename = f'tsne_{split}_features.png'
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        print(f"Saved plot for {split.upper()} to: {output_filename}")
        plt.close()