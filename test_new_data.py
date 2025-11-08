import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score, auc, precision_recall_curve, roc_curve
import os
import gc
import matplotlib.pyplot as plt
import pickle  # 用于保存/加载 encoders

# 导入您的 config, main, model
from config import DEVICE, N_AA, AMINO_ACID_MAPPING, TCR_COLS, PMHC_COLS
from main import one_hot_encode_sequence, prepare_attribute_encoders
# 导入所有模型类（从 model.py）
from model import (
    ERGOIIPredictor, NetTCR2Predictor, ImRexPredictor, DLpTCRPredictor, pMTnetPredictor, DeepTCRPredictor,
    TITANPredictor, PRIME2Predictor, TEINetPredictor, PanPepTCRPredictor, TEIMPredictor, PISTEPredictor,
    MixTCRpredPredictor, TPepRetPredictor, UniPMTPredictor, UnifyImmunPredictor, TCRBaggerPredictor, DeepAntigenPredictor
)

def prepare_new_dataset(new_csv_path, max_cdr3_len, max_epitope_len, attr_encoders, attr_dims, label_threshold=50.0):
    """
    加载新数据集 CSV，预处理为 X_new (features), Y_new (labels if available）。
    """
    df_new = pd.read_csv(new_csv_path)
    df_new['MHC'] = df_new['MHC'].apply(lambda x: str(x).split(',')[0].strip())

    def encode_attr_safe(col_name: str, value: str):
        to_int = attr_encoders.get(col_name, {})
        dim = attr_dims.get(col_name, 0)
        encoded = np.zeros(dim, dtype=np.float32)
        if value in to_int:
            encoded[to_int[value]] = 1.0
        return encoded

    X_new = []
    Y_new = []

    LABEL_COLUMN_NAME = 'Label'
    
    if LABEL_COLUMN_NAME not in df_new.columns:
        print(f"❌ 严重错误: 列 '{LABEL_COLUMN_NAME}' 未在 {new_csv_path} 中找到。")
        print("--- 请打开 test_new_data.py 并编辑 'prepare_new_dataset' 函数，")
        print("--- 将 '!!YOUR_LABEL_COLUMN_NAME_HERE!!' 替换为正确的标签列名 ---")
        # 停止执行，而不是返回空数组
        raise KeyError(f"列 '{LABEL_COLUMN_NAME}' 未在 {new_csv_path} 中找到。")

    for _, row in df_new.iterrows():
        # 序列和属性（适应新列名）
        cdr3_encoded = one_hot_encode_sequence(str(row['CDR3_beta']), max_cdr3_len) if pd.notna(row['CDR3_beta']) else np.zeros(max_cdr3_len * N_AA)
        epitope_encoded = one_hot_encode_sequence(str(row['Epitope_norm']), max_epitope_len) if pd.notna(row['Epitope_norm']) else np.zeros(max_epitope_len * N_AA)
        v_encoded = encode_attr_safe('V_beta', str(row['V_beta']))
        j_encoded = encode_attr_safe('J_beta', str(row['J_beta']))
        mhc_encoded = encode_attr_safe('MHC', str(row['MHC']))
        
        feature_vector = np.concatenate([cdr3_encoded, epitope_encoded, v_encoded, j_encoded, mhc_encoded])
        X_new.append(feature_vector)
        
        # 直接读取标签列，并转换为整数 (0 或 1)
        try:
            label = int(float(row[LABEL_COLUMN_NAME])) if pd.notna(row[LABEL_COLUMN_NAME]) else 0
        except ValueError:
            print(f"警告: 无法将 '{LABEL_COLUMN_NAME}' 值 '{row[LABEL_COLUMN_NAME]}' 转换为数字，已设为 0")
            label = 0
            
        Y_new.append(label)

    X_new = np.array(X_new)
    Y_new = np.array(Y_new)
    
    # 添加检查，确保我们现在有两个类
    unique_labels, counts = np.unique(Y_new, return_counts=True)
    print(f"新数据集样本数: {len(X_new)}")
    print(f"标签分布: {dict(zip(unique_labels, counts))}")
    if len(unique_labels) < 2 and len(X_new) > 0: # 仅在有数据但标签单一时警告
        print("❌ 警告: 加载后，标签中仍然只有一个类别。AUC/AUPRC 将无法计算。")
        print(f"--- 请再次检查您的 CSV 文件和 '{LABEL_COLUMN_NAME}' 列的内容。 ---")
    elif len(X_new) == 0:
         print("❌ 警告: 未加载到任何数据，X_new 为空。")
    
    return X_new, Y_new

# 您的 test_model 函数（从之前修改，确保支持批处理）
def test_model(models, X_test, Y_test, model_name, dataset_name):
    if not isinstance(models, list):
        models = [models]

    dataset_folder = f"{dataset_name}/"
    os.makedirs(dataset_folder, exist_ok=True)
    
    batch_size = 64
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
    test_dataset = TensorDataset(X_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    with torch.no_grad():
        torch.cuda.empty_cache()
        probs = []
        for batch in test_loader:
            X_batch = batch[0]
            batch_probs = []
            for model in models:
                model.to(DEVICE)
                model.eval()
                prob = model(X_batch).squeeze().cpu().numpy()
                batch_probs.append(prob)
            avg_prob = np.mean(batch_probs, axis=0)
            probs.append(avg_prob)
        Y_pred_proba = np.concatenate(probs)
    
    Y_pred_class = (Y_pred_proba >= 0.5).astype(int)
    Y_test = Y_test.astype(int)

    try:
        auc_roc = roc_auc_score(Y_test, Y_pred_proba)
        precision, recall, _ = precision_recall_curve(Y_test, Y_pred_proba)
        auprc = auc(recall, precision)
        
        print(f"AUC (ROC): {auc_roc:.4f}")
        print(f"PRAUC: {auprc:.4f}")
        
        # 保存预测
        results_df = pd.DataFrame({
            'Y_True': Y_test,
            'Y_Predicted_Proba': Y_pred_proba,
            'Y_Predicted_Class': Y_pred_class
        })
        predictions_filename = f"{dataset_folder}{model_name}_predictions.csv"
        results_df.to_csv(predictions_filename, index=False)
        
        # 保存 metrics
        metrics_df = pd.DataFrame({
            'Model': [model_name],
            'AUC_ROC': [auc_roc],
            'PRAUC': [auprc]
        })
        metrics_filename = f"{dataset_folder}{model_name}_metrics.csv"
        metrics_df.to_csv(metrics_filename, index=False)
        
        # 保存 ROC 数据
        fpr, tpr, _ = roc_curve(Y_test, Y_pred_proba)
        roc_df = pd.DataFrame({'FPR': fpr, 'TPR': tpr})
        roc_df.to_csv(f"{dataset_folder}{model_name}_roc_data.csv", index=False)
        
        # 保存 PR 数据
        pr_df = pd.DataFrame({'Precision': precision, 'Recall': recall})
        pr_df.to_csv(f"{dataset_folder}{model_name}_pr_data.csv", index=False)
        
    except ValueError as e:
        print(f"评估失败：{e}。如果无标签，只保存预测。")
        # 只保存预测
        results_df = pd.DataFrame({'Y_Predicted_Proba': Y_pred_proba})
        results_df.to_csv(f"{dataset_folder}{model_name}_predictions.csv", index=False)

# 修改后的绘图函数（从您的原始代码优化，添加随机基线；完善标题）
def plot_all_models_curves(model_names, dataset_name):
    """
    加载所有模型的 ROC/PR 数据和 metrics，绘制 AUROC 和 AUPRC 曲线。
    新增: 标签显示性能值；添加随机基线；完善标题。
    """
    
    dataset_folder = f"{dataset_name}/"
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan', 'magenta', 'yellow']  # 自定义颜色
    
    for i, model_name in enumerate(model_names):
        try:
            # 加载 ROC 数据
            roc_path = f"{dataset_folder}{model_name}_roc_data.csv"
            roc_df = pd.read_csv(roc_path)
            fpr, tpr = roc_df['FPR'], roc_df['TPR']
            
            # 加载 metrics 获取 AUROC
            metrics_path = f"{dataset_folder}{model_name}_metrics.csv"
            metrics_df = pd.read_csv(metrics_path)
            auc_value = metrics_df['AUC_ROC'].iloc[0]
            
            # AUROC 标签: 模型名 + 值
            label_b = f"{model_name} ({auc_value:.3f})"
            ax1.plot(fpr, tpr, color=colors[i % len(colors)], label=label_b, linewidth=2)
            
            # 加载 PR 数据
            pr_path = f"{dataset_folder}{model_name}_pr_data.csv"
            pr_df = pd.read_csv(pr_path)
            precision, recall = pr_df['Precision'], pr_df['Recall']
            
            # AUPRC 标签: 模型名 + 值
            auprc_value = metrics_df['PRAUC'].iloc[0]
            label_c = f"{model_name} ({auprc_value:.3f})"
            ax2.plot(recall, precision, color=colors[i % len(colors)], label=label_c, linewidth=2)
            
        except FileNotFoundError as e:
            print(f"警告：{model_name} 的文件未找到 ({e})，跳过。")
        except KeyError as e:
            print(f"警告：{model_name} CSV 缺少列 ({e})，跳过。")
        except Exception as e:
            print(f"警告：{model_name} 处理失败 ({e})，跳过。")
    
    # 添加随机基线 (AUROC: y=x, 0.500; AUPRC: 在绘图函数最前面读取任一模型的 predictions.csv，算正样本率)
    pred_path = f"{dataset_folder}{model_names[0]}_predictions.csv"
    if os.path.exists(pred_path):
      df = pd.read_csv(pred_path)
      if 'Y_True' in df.columns:
        positive_rate = df['Y_True'].mean()
    else:
      positive_rate = 0.5  # 保底值
    
    
    ax1.plot([0, 1], [0, 1], color='gray', linestyle='--', label='Random (0.500)', linewidth=1.5)
    # 画 AUPRC 随机基线
    ax2.axhline(y=positive_rate, color='gray', linestyle='--',
            label=f'Random ({positive_rate:.3f})', linewidth=1.5)
    
    # AUROC 面板
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title(f'AUROC curves for all models on {dataset_name} dataset')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # AUPRC 面板
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title(f'AUPRC curves for all models on {dataset_name} dataset')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存文件名
    plot_filename = f"{dataset_name}_all_models_roc_pr.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"✅ 曲线图已保存为: {plot_filename}")



if __name__ == '__main__':
    # --- 1. 定义训练路径和模型类 ---
    train_paths = {
        'PROCESSED_IEDB': '/home/dengyang/code/My_code/data/processed_iedb.csv',
        'PROCESSED_MCPAS': '/home/dengyang/code/My_code/data/processed_mcpas.csv',
        'PROCESSED_VDJDB': '/home/dengyang/code/My_code/data/processed_vdjdb.csv'
    }
    
    model_classes = {
        'ERGO-II': ERGOIIPredictor,
        'NetTCR-2.0': NetTCR2Predictor,
        'ImRex': ImRexPredictor,
        'DLpTCR': DLpTCRPredictor,
        'pMTnet': pMTnetPredictor,
        'DeepTCR': DeepTCRPredictor,
        'TITAN': TITANPredictor,
        'PRIME2.0': PRIME2Predictor,
        'TEINet': TEINetPredictor,
        'PanPep': PanPepTCRPredictor,
        'TEIM': TEIMPredictor,
        'PISTE': PISTEPredictor,
        'MixTCRpred': MixTCRpredPredictor,
        'TPepRet': TPepRetPredictor,
        'UniPMT': UniPMTPredictor,
        'UnifyImmun': UnifyImmunPredictor,
        'TCRBagger': TCRBaggerPredictor,
        'DeepAntigen': DeepAntigenPredictor
    }
    
    # --- 2. 加载或创建训练配置 (修正版) ---
    print("--- 正在加载或创建训练配置 (Encoders, Dims, Max Lens) ---")
    train_configs = {}
    
    for train_name, train_path in train_paths.items():
        print(f"处理训练配置: {train_name}")
        config_dir = f"saved_models/{train_name}/"
        os.makedirs(config_dir, exist_ok=True)
        
        encoders_path = f"{config_dir}attr_encoders.pkl"
        dims_path = f"{config_dir}attr_dims.pkl"
        attr_dim_path = f"{config_dir}attr_dim.pkl"
        max_lens_path = f"{config_dir}max_lens.pkl"
        
        # 检查所有文件是否存在 (执行步骤1后，这里应该会失败并进入 else)
        if all(os.path.exists(p) for p in [encoders_path, dims_path, attr_dim_path, max_lens_path]):
            print(f"从 .pkl 文件加载配置: {train_name}")
            with open(encoders_path, 'rb') as f: attr_encoders = pickle.load(f)
            with open(dims_path, 'rb') as f: attr_dims = pickle.load(f)
            with open(attr_dim_path, 'rb') as f: attr_dim = pickle.load(f)
            with open(max_lens_path, 'rb') as f: max_lens = pickle.load(f)
            max_cdr3_len, max_epitope_len = max_lens['max_cdr3_len'], max_lens['max_epitope_len']
        
        else:
            print(f"未找到 .pkl，从 {train_path} 重新计算并保存配置...")
            try:
                train_df = pd.read_csv(train_path, dtype=str, low_memory=False)
                
                # --- 关键修正点: 添加 MHC 清洗 ---
                print("正在清洗 MHC 列以匹配 main.py...")
                train_df['MHC'] = train_df['MHC'].apply(lambda x: str(x).split(',')[0].strip())
                # --- 修正结束 ---

                # 关键: 过滤掉用于计算配置的空值行
                train_df = train_df.dropna(subset=['CDR3_beta', 'Epitope', 'V_beta', 'J_beta', 'MHC'])
                
                attr_cols = ['V_beta', 'J_beta', 'MHC']
                _, attr_dims, attr_encoders = prepare_attribute_encoders(train_df, attr_cols)
                
                max_cdr3_len = train_df['CDR3_beta'].astype(str).apply(len).max()
                max_epitope_len = train_df['Epitope'].astype(str).apply(len).max()
                attr_dim = sum(attr_dims.values())
                
                print(f"计算得到: MaxCDR3={max_cdr3_len}, MaxEpi={max_epitope_len}, AttrDim={attr_dim}")
                
                # 保存配置
                with open(encoders_path, 'wb') as f: pickle.dump(attr_encoders, f)
                with open(dims_path, 'wb') as f: pickle.dump(attr_dims, f)
                with open(attr_dim_path, 'wb') as f: pickle.dump(attr_dim, f)
                with open(max_lens_path, 'wb') as f: pickle.dump({'max_cdr3_len': max_cdr3_len, 'max_epitope_len': max_epitope_len}, f)
                print(f"配置已保存到: {config_dir}")
                
            except Exception as e:
                print(f"❌ 错误: 无法从 {train_path} 生成配置. {e}")
                continue 

        train_configs[train_name] = {
            'max_cdr3_len': int(max_cdr3_len),
            'max_epitope_len': int(max_epitope_len),
            'attr_encoders': attr_encoders,
            'attr_dims': attr_dims,
            'attr_dim': int(attr_dim)
        }
    print("--- 所有训练配置加载完毕 ---")

    # --- 3. 定义新数据集 ---
    new_datasets = {
        'MUTATION_UNSEEN': '/home/dengyang/code/My_code/filtered_mutation_unseen_balanced_strict.csv',
        'MUTATION_MOUSE_UNSEEN': '/home/dengyang/code/My_code/filtered_mutation_mouse_unseen_balanced_strict.csv'
    }
    
    # --- 4. 评估循环逻辑 (与上次相同) ---
    
    # [外循环] 遍历新数据集
    for new_name, new_path in new_datasets.items():
        
        # [中循环] 遍历用于训练模型的配置 (IEDB, McPAS, VDJdb)
        for train_name, config in train_configs.items():
            
            new_dataset_folder = f"{new_name}_from_{train_name}"
            os.makedirs(new_dataset_folder, exist_ok=True)
            
            print(f"\n{'='*80}")
            print(f"开始评估: {new_name} (使用 {train_name} 的模型和配置)")
            print(f"输出文件夹: {new_dataset_folder}")
            # 此时的 config['attr_dim'] 应该是正确的 (例如 301)
            print(f"使用配置: MaxCDR3={config['max_cdr3_len']}, MaxEpi={config['max_epitope_len']}, AttrDim={config['attr_dim']}")
            print(f"{'='*80}")

            # [加载和预处理新数据]
            print(f"正在从 {new_path} 加载和预处理新数据...")
            try:
                X_new, Y_new = prepare_new_dataset(
                    new_path, 
                    config['max_cdr3_len'], 
                    config['max_epitope_len'], 
                    config['attr_encoders'], 
                    config['attr_dims']
                )
                print(f"新数据加载完毕，形状 X: {X_new.shape}, Y: {Y_new.shape}")
            except KeyError as e:
                print(f"❌ 严重错误: 在 'prepare_new_dataset' 中发生 KeyError: {e}")
                print("--- 请修正 'prepare_new_dataset' 函数以匹配您的新数据集列名 (例如 'Activation_label') ---")
                continue # 跳过这个训练配置
            except Exception as e:
                print(f"❌ 严重错误: 'prepare_new_dataset' 失败. {e}")
                continue

            models_tested_for_plotting = [] 
            
            # [内循环] 遍历所有模型
            for model_name, model_class in model_classes.items():
                print(f"\n--- 正在测试模型: {model_name} ---")
                
                try:
                    model = model_class(config['max_cdr3_len'], config['max_epitope_len'], config['attr_dim'])
                except Exception as e:
                    print(f"错误: 实例化 {model_name} 失败. {e}")
                    continue
                    
                models_to_test = [] 
                
                if model_name == 'TCRBagger':
                    print("TCRBagger: 正在加载 5 个子模型...")
                    num_bags = 5 
                    all_bags_found = True
                    for i in range(num_bags):
                        sub_model_path = f"saved_models/{train_name}/{model_name}_sub{i}_trained.pt"
                        if os.path.exists(sub_model_path):
                            sub_model = model_class(config['max_cdr3_len'], config['max_epitope_len'], config['attr_dim'])
                            sub_model.load_state_dict(torch.load(sub_model_path, map_location=DEVICE))
                            models_to_test.append(sub_model)
                        else:
                            print(f"警告: {sub_model_path} 未找到。")
                            all_bags_found = False
                    
                    if not all_bags_found or not models_to_test:
                        print(f"警告: 未能加载所有 TCRBagger 子模型，跳过 {model_name}")
                        continue
                        
                else: 
                    save_path = f"saved_models/{train_name}/{model_name}_trained.pt"
                    if os.path.exists(save_path):
                        try:
                            model.load_state_dict(torch.load(save_path, map_location=DEVICE))
                            models_to_test.append(model) 
                        except RuntimeError as e:
                            print(f"❌ 错误: 加载 {save_path} 状态字典失败 (架构不匹配). {e}")
                            print(f"--- 这表明 'attr_dim' ({config['attr_dim']}) 可能仍与模型不匹配 ---")
                            continue
                    else:
                        print(f"警告: {save_path} 未找到，跳过 {model_name}")
                        continue
                
                print(f"正在对 {model_name} (含 {len(models_to_test)} 个实例) 运行评估...")
                try:
                    test_model(
                        models_to_test,
                        X_new, 
                        Y_new, 
                        model_name, 
                        new_dataset_folder 
                    )
                    models_tested_for_plotting.append(model_name)
                    print(f"✅ {model_name} 评估完成。")
                except Exception as e:
                    print(f"❌ 错误: {model_name} 在 test_model 期间失败. {e}")

            # [内循环结束]
            
            # [绘图]
            if models_tested_for_plotting:
                print(f"\n--- G{new_dataset_folder} 绘制性能曲线图 ---")
                try:
                    plot_all_models_curves(models_tested_for_plotting, new_dataset_folder)
                    print(f"✅ 曲线图已保存。")
                except Exception as e:
                    print(f"❌ 错误: 绘制曲线图失败. {e}")
            else:
                print(f"--- {new_dataset_folder} 没有任何模型成功运行，跳过绘图 ---")

        # [中循环结束]
        
        print(f"--- 完成 {new_name} 的所有评估，清理内存 ---")
        gc.collect()
        torch.cuda.empty_cache()

    print("\n=== 所有新数据集评估完成 ===")