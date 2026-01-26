import os
import argparse
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from typing import Dict, Any, List, Tuple

# --- 配置 ---
sns.set_theme(style="white", font_scale=1.2)

# 假设您的文件名为 bbb_logbb_train_features.pkl 等
FILE_PATHS = {
    'train': "bbb_logbb_train_features.pkl",
    'valid': "bbb_logbb_valid_features.pkl",
    'test': "bbb_logbb_test_features.pkl",
}
SPLITS_TO_PLOT = ['train', 'valid', 'test']


# --- 核心函数：数据加载与转换 ---

def normalize_features(X: np.ndarray) -> np.ndarray:
    """强制对特征进行 L2 归一化，模拟单位特征 Z。"""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    # 避免除以零
    norms = np.where(norms == 0, 1e-8, norms)
    return X / norms


def load_and_project_data(file_paths: Dict[str, str]) -> Tuple[pd.DataFrame, bool]:
    """加载 X 特征，进行 L2 归一化，然后执行 3D PCA。"""
    all_features_norm = []
    all_labels = []
    all_splits = []

    print("--- 1. Loading and Normalizing existing X features ---")
    for split_name, path in file_paths.items():
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
                features_X = data['features']
                labels = np.squeeze(data['y_true'])

                # 关键步骤：强制归一化 X -> Z
                features_Z = normalize_features(features_X)

                all_features_norm.append(features_Z)
                all_labels.append(labels)
                all_splits.extend([split_name] * len(labels))
                print(f"Loaded and Normalized {split_name.upper()}: {features_Z.shape[0]} samples.")

        except FileNotFoundError:
            print(f"[ERROR] 文件未找到: {path}. Aborting.")
            return None, False
        except Exception as e:
            print(f"[ERROR] 读取文件 {path} 失败: {e}. Aborting.")
            return None, False

    if not all_features_norm: return None, False

    # 2. 合并数据
    Z_combined = np.concatenate(all_features_norm, axis=0)
    y_combined = np.concatenate(all_labels, axis=0)
    print(f"\nTotal combined normalized samples: {Z_combined.shape[0]}")

    # 3. 执行 PCA 降维到 3D
    print("--- 2. Running PCA to 3D for Spherical Coordinates ---")
    pca = PCA(n_components=3, random_state=42)
    Z_3d = pca.fit_transform(Z_combined)

    # 4. 转换为球面坐标 (经度/纬度, Equirectangular Projection)
    # 重新归一化 3D 结果以确保在单位球体上
    norms = np.linalg.norm(Z_3d, axis=1, keepdims=True)
    Z_spherical = Z_3d / np.where(norms == 0, 1e-8, norms)

    X = Z_spherical[:, 0];
    Y = Z_spherical[:, 1];
    Z = Z_spherical[:, 2]
    phi = np.arcsin(Z)  # 纬度 (Latitude)
    theta = np.arctan2(Y, X)  # 经度 (Longitude)

    # 5. 准备绘图数据
    plot_df = pd.DataFrame({
        'Longitude': np.degrees(theta),
        'Latitude': np.degrees(phi),
        'Label': y_combined.astype(int).astype(str),
        'Split': all_splits
    })
    return plot_df, True


def plot_spherical_map_projection(plot_df: pd.DataFrame, output_path: str = 'spherical_map_projection.png'):
    """绘制单位球面 (Equirectangular Map Projection) 展开图。"""

    print("--- 3. Generating Spherical Map Projection Plot ---")

    plt.figure(figsize=(14, 7))

    # 绘图：按 Label 着色，按 Split 标记样式
    sns.scatterplot(
        x='Longitude', y='Latitude', hue='Label', style='Split', data=plot_df,
        palette={'0': '#3498db', '1': '#e74c3c'},
        s=40, alpha=0.6, linewidth=0.5, edgecolor='w'
    )

    # 坐标轴格式化 (地图网格)
    plt.xlim(-180, 180);
    plt.ylim(-90, 90)
    plt.gca().set_xticks(np.arange(-180, 180 + 1, 30));
    plt.gca().set_yticks(np.arange(-90, 90 + 1, 30))
    plt.gca().axhline(0, color='gray', linestyle='--', linewidth=0.5)
    plt.grid(True, linestyle=':', alpha=0.6)

    plt.title('Latent Features Spherical Map Projection (Forced Normalization)', fontsize=16)
    plt.xlabel('Longitude', fontsize=14);
    plt.ylabel('Latitude', fontsize=14)

    plt.legend(title='Category', bbox_to_anchor=(1.01, 1), loc='upper left', frameon=True)
    sns.despine(trim=True)

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nSuccessfully generated Spherical Map Projection plot: {output_path}")


# --- Execution Entry ---
if __name__ == "__main__":
    plot_df, success = load_and_project_data(FILE_PATHS)

    if success:
        plot_spherical_map_projection(plot_df)