#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
批量运行 test_only_arcmol.py 脚本。
功能：
1. 读取 tasks_csv 获取任务名 (task_name) 和数据目录 (data_dir)。
2. 遍历指定检查点目录 (checkpoints_root) 下，每个任务文件夹内的所有 .pt 文件作为 --bundle，并找到对应的 .pth 文件作为 --ckpt。
3. 对每对文件，执行一次 test_only_arcmol.py 命令，同时传入 --bundle 和 --ckpt。
4. 输出的预测文件以任务名和模型名命名。
5. 包含 --extra_attrs "SMILES,cliff_mol"。
"""

import os
import subprocess
import glob
import pandas as pd
from pathlib import Path
from typing import List, Tuple

# --- 配置参数 ---
# 任务 CSV 文件路径
TASKS_CSV_PATH = "tasks_template_admet.csv"
# test_only_arcmol.py 脚本的路径
TEST_SCRIPT_PATH = "test_only_arcmol.py"
# 模型的根目录，ArcMol 模型文件 (*.pt, *.pth) 应该在这个目录下按任务名组织
CHECKPOINTS_ROOT = "checkpoints"
# 最终结果 CSV 的输出目录
OUTPUT_ROOT = "batch_test_results"
# 固定不变的额外属性
EXTRA_ATTRS = "SMILES"


# -----------------

def get_tasks_from_csv(csv_path: str) -> List[dict]:
    """
    从 CSV 文件中读取任务信息。
    要求 CSV 必须包含 'task_name' 和 'data_dir' 列。
    """
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"[ERROR] 任务文件未找到: {csv_path}")
        return []

    required_cols = ['task_name', 'data_dir']
    if not all(col in df.columns for col in required_cols):
        print(f"[ERROR] 任务 CSV 缺少必需的列 ('task_name' 和 'data_dir')。")
        return []

    return df[required_cols].to_dict('records')


def get_model_bundles(task_name: str, checkpoints_root: str) -> List[Tuple[str, str]]:
    """
    根据任务名，在检查点根目录下查找所有 .pt 文件，并尝试找到对应的 .pth 文件。
    返回 (bundle_path, ckpt_path) 元组列表。
    """
    search_pt_path = Path(checkpoints_root) / task_name / "**" / "*.pt"

    all_pt_files = glob.glob(str(search_pt_path), recursive=True)

    bundles_and_ckpts = []

    for bundle_path in all_pt_files:
        bundle_p = Path(bundle_path)

        # 假设 .pth 文件名与 .pt 文件名主体相同，只是后缀不同。
        # 处理 .bundle.pt 和 .pt 两种情况
        if bundle_p.name.endswith('.bundle.pt'):
            ckpt_filename = bundle_p.name.replace('.bundle.pt', '.pth')
        elif bundle_p.name.endswith('.pt'):
            ckpt_filename = bundle_p.name.replace('.pt', '.pth')
        else:
            continue  # 忽略不匹配的文件

        ckpt_path = bundle_p.parent / ckpt_filename

        if ckpt_path.exists():
            bundles_and_ckpts.append((str(bundle_p), str(ckpt_path)))
        else:
            print(f"[WARN] 任务 {task_name}: Bundle ({bundle_p.name}) 找到，但未找到对应的 CKPT 文件: {ckpt_path}")

    # 仅保留最佳或最终模型（可选过滤）
    # filtered_bundles = [b for b in bundles_and_ckpts if 'best' in Path(b[0]).name.lower() or 'final' in Path(b[0]).name.lower()]
    # if not filtered_bundles:
    #     filtered_bundles = bundles_and_ckpts

    return sorted(bundles_and_ckpts)


def run_test_command(task_info: dict, bundle_path: str, ckpt_path: str, output_root: str):
    """
    构造并执行 test_only_arcmol.py 命令，传入 --bundle 和 --ckpt。
    """
    task_name = task_info['task_name']
    data_dir = task_info['data_dir']

    # 1. 确定输出文件名 (根据任务名和模型名)
    # 使用 ckpt 文件名作为模型标识符
    model_file_name = Path(ckpt_path).name

    # 命名格式: 任务名_模型文件名(去除后缀).csv
    output_filename = f"{task_name}_{model_file_name}".replace('.pth', '.csv').replace('.pt', '.csv')
    output_path = Path(output_root) / output_filename

    # 2. 构造命令
    command = [
        "python", TEST_SCRIPT_PATH,
        "--data_dir", data_dir,
        "--task_name", task_name,
        "--bundle", bundle_path,
        "--ckpt", ckpt_path,  # <<< 新增的 CKPT 参数
        "--save_preds", str(output_path),
        "--extra_attrs", EXTRA_ATTRS
    ]

    print("-" * 50)
    print(f"[TASK] {task_name}")
    print(f"[MODEL] Bundle: {Path(bundle_path).name} | CKPT: {Path(ckpt_path).name}")
    print(f"[OUTPUT] {output_path}")

    # 3. 执行命令
    try:
        # 使用 subprocess.run 执行命令，并捕获输出/错误
        result = subprocess.run(
            command,
            check=True,  # 如果命令返回非零状态码，则抛出异常
            text=True,
            capture_output=True
        )
        print(f"[SUCCESS] {task_name} 测试完成。")
        # 如果命令成功，您可以选择性地打印输出
        # print("--- Command Output ---")
        # print(result.stdout)

    except subprocess.CalledProcessError as e:
        print(f"[FAILURE] {task_name} 命令执行失败，返回码 {e.returncode}。")
        print(f"--- Stderr ---")
        print(e.stderr)
    except FileNotFoundError:
        print(f"[CRITICAL] 脚本未找到: {TEST_SCRIPT_PATH}。请检查路径是否正确。")

    print("-" * 50)


def main():
    # 确保输出目录存在
    Path(OUTPUT_ROOT).mkdir(parents=True, exist_ok=True)

    # 获取所有任务
    tasks = get_tasks_from_csv(TASKS_CSV_PATH)
    if not tasks:
        return

    total_tasks = 0
    total_bundles = 0

    for task_info in tasks:
        task_name = task_info['task_name']

        # 查找该任务下的所有 (bundle, ckpt) 对
        bundles_and_ckpts = get_model_bundles(task_name, CHECKPOINTS_ROOT)

        if not bundles_and_ckpts:
            print(f"[SKIP] 任务 {task_name}: 在 {CHECKPOINTS_ROOT}/{task_name} 目录下未找到任何 (bundle.pt, ckpt.pth) 对。")
            continue

        total_bundles += len(bundles_and_ckpts)
        total_tasks += 1

        for bundle, ckpt in bundles_and_ckpts:
            run_test_command(task_info, bundle, ckpt, OUTPUT_ROOT)

    print("\n" + "=" * 50)
    print(f"✅ 批量测试完成！")
    print(f"处理任务数: {total_tasks}")
    print(f"测试模型总数: {total_bundles}")
    print(f"结果保存在目录: {OUTPUT_ROOT}")
    print("=" * 50)


if __name__ == "__main__":
    main()
