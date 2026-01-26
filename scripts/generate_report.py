import os
import glob
import pandas as pd
import numpy as np
from sklearn.metrics import (
    roc_auc_score, matthews_corrcoef, f1_score, accuracy_score,
    mean_squared_error, mean_absolute_error, r2_score
)

# ================= 配置部分 =================
RESULTS_DIR = "batch_test_results"       # 结果文件夹
TASK_TEMPLATE_PATH = "tasks_template_admet.csv"  # 任务类型模板
OUTPUT_FILE = "my_model_summary.csv"     # 输出文件
# ===========================================

def calculate_cls_metrics(y_true, y_pred_prob):
    """计算分类指标 (输入是真实标签和预测概率)"""
    try:
        # 1. AUC (如果有多个类别才算，否则 NaN)
        if len(np.unique(y_true)) > 1:
            auc = roc_auc_score(y_true, y_pred_prob)
        else:
            auc = np.nan

        # 2. 其他指标需要 0/1 的预测标签 (阈值 0.5)
        # 注意：这里假设 prob > 0.5 为正例
        y_pred_bin = (np.array(y_pred_prob) > 0.5).astype(int)
        
        mcc = matthews_corrcoef(y_true, y_pred_bin)
        f1 = f1_score(y_true, y_pred_bin, zero_division=0)
        acc = accuracy_score(y_true, y_pred_bin)
        
        return {
            "Test_AUC": auc,
            "Test_MCC": mcc,
            "Test_F1": f1,
            "Test_ACC": acc,
            # 回归指标留空
            "Test_RMSE": np.nan,
            "Test_MAE": np.nan,
            "Test_R2": np.nan
        }
    except Exception as e:
        print(f"  [Error] Cls metrics calculation failed: {e}")
        return {}

def calculate_reg_metrics(y_true, y_pred):
    """计算回归指标 (输入是真实值和预测值)"""
    try:
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        return {
            # 分类指标留空
            "Test_AUC": np.nan,
            "Test_MCC": np.nan,
            "Test_F1": np.nan,
            "Test_ACC": np.nan,
            "Test_RMSE": rmse,
            "Test_MAE": mae,
            "Test_R2": r2
        }
    except Exception as e:
        print(f"  [Error] Reg metrics calculation failed: {e}")
        return {}

def main():
    # 1. 读取任务类型字典
    if not os.path.exists(TASK_TEMPLATE_PATH):
        print(f"错误: 找不到 {TASK_TEMPLATE_PATH}")
        return
    
    tasks_df = pd.read_csv(TASK_TEMPLATE_PATH)
    task_types = dict(zip(tasks_df['task_name'], tasks_df['task_type']))
    
    # 2. 遍历结果文件
    results_list = []
    
    if not os.path.exists(RESULTS_DIR):
        print(f"错误: 找不到目录 {RESULTS_DIR}")
        return

    csv_files = glob.glob(os.path.join(RESULTS_DIR, "*.csv"))
    print(f"找到 {len(csv_files)} 个文件，开始处理...\n")
    
    for file_path in csv_files:
        file_name = os.path.basename(file_path)
        
        # --- A. 匹配任务名 ---
        matched_task = None
        longest_match_len = 0
        for t_name in task_types.keys():
            if t_name in file_name:
                if len(t_name) > longest_match_len:
                    matched_task = t_name
                    longest_match_len = len(t_name)
        
        if not matched_task:
            print(f"⚠️ [跳过] 无法识别任务名: {file_name}")
            continue
            
        t_type = task_types[matched_task]
        
        # --- B. 读取并提取特定列 ---
        try:
            df = pd.read_csv(file_path)
            
            # 根据你提供的格式，强制指定列名
            if t_type == 'reg':
                # 回归任务：找 'y_true' 和 'y_pred'
                required_cols = ['y_true', 'y_pred']
                if not all(col in df.columns for col in required_cols):
                    print(f"❌ [错误] {matched_task} (reg) 缺少必要列 {required_cols}。现有列: {df.columns.tolist()}")
                    continue
                y_true = df['y_true']
                y_pred = df['y_pred']
                
            elif t_type == 'cls':
                # 分类任务：找 'y_true' 和 'prob'
                required_cols = ['y_true', 'prob']
                if not all(col in df.columns for col in required_cols):
                    print(f"❌ [错误] {matched_task} (cls) 缺少必要列 {required_cols}。现有列: {df.columns.tolist()}")
                    continue
                y_true = df['y_true']
                y_pred = df['prob']
            
            else:
                print(f"⚠️ [跳过] 未知任务类型: {t_type}")
                continue

            # 去除 NaN (防止最后有空行)
            mask = y_true.notna() & y_pred.notna()
            y_true = y_true[mask]
            y_pred = y_pred[mask]

            if len(y_true) == 0:
                print(f"⚠️ [跳过] {matched_task}: 有效数据为 0")
                continue

            # --- C. 计算指标 ---
            metrics = {
                "task_name": matched_task,
                "task_type": t_type,
                "filename": file_name
            }
            
            if t_type == 'cls':
                metrics.update(calculate_cls_metrics(y_true, y_pred))
            elif t_type == 'reg':
                metrics.update(calculate_reg_metrics(y_true, y_pred))
                
            results_list.append(metrics)
            print(f"✅ {matched_task:<20} | {t_type:<3} | OK")
            
        except Exception as e:
            print(f"❌ [异常] 处理 {file_name} 失败: {e}")

    # 3. 保存结果
    if results_list:
        final_df = pd.DataFrame(results_list)
        
        # 整理列顺序
        cols = ['task_name', 'task_type', 
                'Test_AUC', 'Test_MCC', 'Test_F1', 'Test_ACC', 
                'Test_RMSE', 'Test_MAE', 'Test_R2', 
                'filename']
        # 只保留存在的列
        cols = [c for c in cols if c in final_df.columns]
        final_df = final_df[cols]
        
        # 按任务名排序
        final_df.sort_values(by='task_name', inplace=True)

        final_df.to_csv(OUTPUT_FILE, index=False)
        print("\n" + "="*50)
        print(f"处理完成！报告已生成: {OUTPUT_FILE}")
        print("="*50)
        # 预览
        print(final_df[['task_name', 'task_type', 'Test_AUC', 'Test_RMSE']].head().to_string())
    else:
        print("\n❌ 未生成任何结果。")

if __name__ == "__main__":
    main()

