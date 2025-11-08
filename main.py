# -*- coding: utf-8 -*-

# main.py

import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from typing import List, Tuple
import os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, auc
import gc  # 新增: 垃圾回收

# 导入配置、模型和训练/测试函数
from config import SEED, N_AA, TCR_COLS, PMHC_COLS, AMINO_ACID_MAPPING, TRAIN_PARAMS
from model import (
    DeepTCRPredictor, pMTnetPredictor, PRIME2Predictor,
    ERGOIIPredictor, NetTCR2Predictor, ImRexPredictor, TEIMPredictor, MixTCRpredPredictor,PRIME2Predictor, UnifyImmunPredictor,UniPMTPredictor,
    DeepAntigenPredictor,PanPepTCRPredictor,TITANPredictor,PISTEPredictor,TPepRetPredictor,TCRBaggerPredictor,TEINetPredictor  ,DLpTCRPredictor,
    train_model, test_model
)

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:1024'
torch.backends.cudnn.benchmark = False  # 固定分配，避免碎片
gc.collect()
torch.cuda.empty_cache()
# 设置随机种子
np.random.seed(SEED)
torch.manual_seed(SEED)


# --- 1. 数据预处理辅助函数 ---

def load_and_preprocess_data(file_name="iedb.csv"):
    """加载数据，处理 MHC 列。"""
    if not os.path.exists(file_name):
        raise FileNotFoundError(f"找不到文件: {file_name}，请确保它位于脚本的同一目录下。")
    df = pd.read_csv(file_name)
    df['MHC'] = df['MHC'].apply(lambda x: str(x).split(',')[0].strip())
    return df

def one_hot_encode_sequence(sequence: str, max_len: int, mapping: dict = AMINO_ACID_MAPPING) -> np.ndarray:
    """对序列进行独热编码。"""
    L = len(sequence)
    encoded = np.zeros((max_len, N_AA), dtype=np.float32)
    for i in range(min(L, max_len)):
        aa = sequence[i].upper()
        if aa in mapping:
            encoded[i, mapping[aa]] = 1.0
    return encoded.flatten()

def prepare_attribute_encoders(df: pd.DataFrame, columns: List[str]) -> Tuple[callable, dict, dict]:
    """准备分类属性编码器和维度。"""
    encoders = {}
    dims = {}
    for col in columns:
        classes = sorted(df[col].dropna().unique().tolist())
        to_int = {cls: i for i, cls in enumerate(classes)}
        encoders[col] = to_int
        dims[col] = len(classes)
        
    def encode_attr(col_name: str, value: str) -> Tuple[np.ndarray, int]:
        dim = dims.get(col_name, 0)
        to_int = encoders.get(col_name, {})
        encoded = np.zeros(dim, dtype=np.float32)
        if value in to_int:
            encoded[to_int[value]] = 1.0
        return encoded, dim
    return encode_attr, dims, encoders


# --- 2. 特征工程和数据生成逻辑 ---

def generate_tcr_pmhc_dataset(df):
    """生成 TCR-pMHC 结合预测任务的数据集。"""
    
    df_tcr = df.dropna(subset=TCR_COLS + PMHC_COLS).copy()
    
    # 1. 计算维度
    MAX_CDR3_LEN = df_tcr['CDR3_beta'].apply(len).max()
    MAX_EPITOPE_LEN = df_tcr['Epitope'].apply(len).max()

    attr_cols = ['V_beta', 'J_beta', 'MHC']
    encode_attr, dims, _ = prepare_attribute_encoders(df_tcr, attr_cols)
    ATTR_DIM = dims.get('V_beta', 0) + dims.get('J_beta', 0) + dims.get('MHC', 0)
    
    print(f"阳性样本数: {len(df_tcr)}, 最大CDR3: {MAX_CDR3_LEN}, 最大Epitope: {MAX_EPITOPE_LEN}, 属性维数: {ATTR_DIM}")

    # 2. 阳性样本编码
    X_pos = []
    for _, row in df_tcr.iterrows():
        cdr3_encoded = one_hot_encode_sequence(row['CDR3_beta'], MAX_CDR3_LEN)
        epitope_encoded = one_hot_encode_sequence(row['Epitope'], MAX_EPITOPE_LEN)
        v_encoded, _ = encode_attr('V_beta', row['V_beta'])
        j_encoded, _ = encode_attr('J_beta', row['J_beta'])
        mhc_encoded, _ = encode_attr('MHC', row['MHC'])
        
        feature_vector = np.concatenate([cdr3_encoded, epitope_encoded, v_encoded, j_encoded, mhc_encoded])
        X_pos.append(feature_vector)
    
    X_pos = np.array(X_pos)
    Y_pos = np.ones(len(X_pos), dtype=np.int32)

    # 3. 负样本生成 (置换法)
    N_pos = len(df_tcr)
    X_neg = []
    tcr_data = df_tcr[TCR_COLS].values.tolist()
    pmhc_data = df_tcr[PMHC_COLS].values.tolist()
    positive_pairs = set(tuple(row) for row in df_tcr[TCR_COLS + PMHC_COLS].values)

    while len(X_neg) < N_pos:
        idx_tcr = np.random.randint(N_pos)
        idx_pmhc = np.random.randint(N_pos)
        
        tcr_cdr3, tcr_v, tcr_j = tcr_data[idx_tcr]
        pmhc_epitope, pmhc_mhc = pmhc_data[idx_pmhc]
        
        if (tcr_cdr3, tcr_v, tcr_j, pmhc_epitope, pmhc_mhc) in positive_pairs:
             continue 

        cdr3_encoded = one_hot_encode_sequence(tcr_cdr3, MAX_CDR3_LEN)
        epitope_encoded = one_hot_encode_sequence(pmhc_epitope, MAX_EPITOPE_LEN)
        
        v_encoded, _ = encode_attr('V_beta', tcr_v)
        j_encoded, _ = encode_attr('J_beta', tcr_j)
        mhc_encoded, _ = encode_attr('MHC', pmhc_mhc)
        
        feature_vector = np.concatenate([cdr3_encoded, epitope_encoded, v_encoded, j_encoded, mhc_encoded])
        X_neg.append(feature_vector)

    X_neg = np.array(X_neg)
    X_neg = X_neg[:len(X_pos)]
    Y_neg = np.zeros(len(X_neg), dtype=np.int32)
    
    # 4. 合并和划分
    X_data = np.concatenate([X_pos, X_neg], axis=0)
    Y_data = np.concatenate([Y_pos, Y_neg], axis=0)

    X_train, X_test, Y_train, Y_test = train_test_split(
        X_data, Y_data, test_size=0.2, random_state=SEED, stratify=Y_data
    )
    print(f"总训练集样本数: {len(X_train)}, 总测试集样本数: {len(X_test)}")
    
    return X_train, X_test, Y_train, Y_test, MAX_CDR3_LEN, MAX_EPITOPE_LEN, ATTR_DIM

# --- 3. 模型选择和执行函数 ---

def run_model_comparison(df, model_name: str, dataset_name: str = None):
    """
    根据模型名称，选择并运行 TCR-pMHC 结合预测模型。
    """
    print(f"\n{'='*60}\n正在运行模型: {model_name} (数据集: {dataset_name})\n{'='*60}")
    
    # 1. 数据准备 (所有TCR-pMHC模型使用相同的数据)
    X_train, X_test, Y_train, Y_test, MAX_CDR3_LEN, MAX_EPITOPE_LEN, ATTR_DIM = generate_tcr_pmhc_dataset(df)

    
    if 'VDJDB' in dataset_name.upper() or 'IEDB' in dataset_name.upper():  # 对于 VDJdb or IEDB，调整 lr 和 epochs
        custom_params = TRAIN_PARAMS.copy()
        custom_params['learning_rate'] = 1e-4  # 降低 lr 以适应稀疏数据
        custom_params['epochs'] = 80  # 增加 epochs 以允许收敛
    else:  # McPAS 或其他，保持原值
        custom_params = TRAIN_PARAMS

    
    # 2. 模型实例化    
    if model_name == 'DeepTCR':
        # DeepTCR 风格模型 (CNN + FC)
        model = DeepTCRPredictor(MAX_CDR3_LEN, MAX_EPITOPE_LEN, ATTR_DIM)

    elif model_name == 'pMTnet':
        # pMTnet 模型 (Atchley CNN-TCR + LSTM-Epi)
        model = pMTnetPredictor(MAX_CDR3_LEN, MAX_EPITOPE_LEN, ATTR_DIM)
    
    elif model_name == 'ERGO-II': 
        # ERGO-II 模型 (LSTM 序列编码)
        model = ERGOIIPredictor(MAX_CDR3_LEN, MAX_EPITOPE_LEN, ATTR_DIM)

    elif model_name == 'NetTCR-2.0': 
        # NetTCR-2.0 模型 (Multi-Kernel CNN)
        model = NetTCR2Predictor(MAX_CDR3_LEN, MAX_EPITOPE_LEN, ATTR_DIM)

    elif model_name == 'DLpTCR': 
        # DLpTCR 模型 (Ensemble: FULL MLP + CNN + ResNet1D)
        model = DLpTCRPredictor(MAX_CDR3_LEN, MAX_EPITOPE_LEN, ATTR_DIM)

    elif model_name == 'PRIME2.0': 
        # PRIME2.0 模型 (Transformer 编码 + Attention Pooling)
        model = PRIME2Predictor(MAX_CDR3_LEN, MAX_EPITOPE_LEN, ATTR_DIM)
        
    elif model_name == 'ImRex': 
        # ImRex 模型 (Interaction Map 2D CNN)
        model = ImRexPredictor(MAX_CDR3_LEN, MAX_EPITOPE_LEN, ATTR_DIM)

    elif model_name == 'TEIM': 
        # TEIM 模型 (Interaction Map 2D ResNet)
        model = TEIMPredictor(MAX_CDR3_LEN, MAX_EPITOPE_LEN, ATTR_DIM)

    elif model_name == 'MixTCRpred': 
        # MixTCRpred 风格模型 (Transformer 编码器)
        model = MixTCRpredPredictor(MAX_CDR3_LEN, MAX_EPITOPE_LEN, ATTR_DIM)

    elif model_name == 'UnifyImmun': 
        # UnifyImmun 模型 (Self-Attention)
        model = UnifyImmunPredictor(MAX_CDR3_LEN, MAX_EPITOPE_LEN, ATTR_DIM)

    elif model_name == 'UniPMT': 
        # UniPMT 模型 (CNN Enc + PM/MT MLP 融合)
        model = UniPMTPredictor(MAX_CDR3_LEN, MAX_EPITOPE_LEN, ATTR_DIM)

    elif model_name == 'DeepAntigen': 
        # DeepAntigen 模型 (1D-GCN + SuperNode Attn)
        model = DeepAntigenPredictor(MAX_CDR3_LEN, MAX_EPITOPE_LEN, ATTR_DIM)

    elif model_name == 'PanPep': 
        # PanPep_TCR 模型 (Atchley Joint Matrix + Self-Attn + 2D CNN)
        model = PanPepTCRPredictor(MAX_CDR3_LEN, MAX_EPITOPE_LEN, ATTR_DIM)

    elif model_name == 'TITAN':
        # TITAN 模型 (Bi-LSTM + Bimodal Attention)
        model = TITANPredictor(MAX_CDR3_LEN, MAX_EPITOPE_LEN, ATTR_DIM)

    elif model_name == 'PISTE':
        # PISTE 模型 (Conv-Encoder + Transformer-Decoder)
        model = PISTEPredictor(MAX_CDR3_LEN, MAX_EPITOPE_LEN, ATTR_DIM)

    elif model_name == 'TPepRet':
        # TPepRet 模型 (Transformer 编码 + Cross-Attn)
        model = TPepRetPredictor(MAX_CDR3_LEN, MAX_EPITOPE_LEN, ATTR_DIM)        

    elif model_name == 'TCRBagger': 
        # TCRBagger 基础模型 (Flatten-CNN + MIL Attention)
        model = TCRBaggerPredictor(MAX_CDR3_LEN, MAX_EPITOPE_LEN, ATTR_DIM)

    elif model_name == 'TEINet': 
        # TEINet 模型 (LSTM 编码 + MLP 投影)
        model = TEINetPredictor(MAX_CDR3_LEN, MAX_EPITOPE_LEN, ATTR_DIM)
                        
    else:
        # --- 修正: 更新了错误提示中的模型列表 ---
        raise ValueError(f"未知模型名称: {model_name}。请检查您的模型列表。")
    
    # 获取最终模型输入维度，用于打印
    if hasattr(model, 'seq_feature_dim'):
        total_input_dim = model.seq_feature_dim
    else:
        total_input_dim = "N/A (复杂架构)"

    print(f"【{model_name}】 模型实例化成功，最终分类器输入维: {total_input_dim}")

    
    # 3. 训练模型 - 对于TCRBagger，使用bagging训练
    
    save_dir = f"saved_models/{dataset_name}/"
    os.makedirs(save_dir, exist_ok=True)  # 创建文件夹
    
    
    if model_name == 'TCRBagger':
        # Bagging: 训练多个实例
        num_bags = 5  # 原始bagging数量示例
        models = [TCRBaggerPredictor(MAX_CDR3_LEN, MAX_EPITOPE_LEN, ATTR_DIM) for _ in range(num_bags)]
        trained_models = []
        for i, sub_model in enumerate(models):
            # 子采样训练数据 (bagging)
            indices = np.random.choice(len(X_train), size=int(0.8 * len(X_train)), replace=True)
            X_sub = X_train[indices]
            Y_sub = Y_train[indices]
            trained_sub = train_model(sub_model, X_sub, Y_sub, custom_params)
            # 保存每个 sub_model
            torch.save(trained_sub.state_dict(), f"{save_dir}{model_name}_sub{i}_trained.pt")
            trained_models.append(trained_sub)
        # 测试时平均预测
        test_model(trained_models, X_test, Y_test, model_name, dataset_name)  # 修改test_model支持list
    else:
        trained_model = train_model(model, X_train, Y_train, custom_params)
        # 保存单个模型
        torch.save(trained_model.state_dict(), f"{save_dir}{model_name}_trained.pt")
        # 4. 测试模型
        test_model(trained_model, X_test, Y_test, model_name,dataset_name)


    if model_name == 'TCRBagger':
        del models, trained_models  # 删除 bagging 相关变量
    else:
        del model, trained_model  # 删除标准变量
        
        
    torch.cuda.empty_cache()
    gc.collect()  # 导入 gc 在顶部
    print(f"[{model_name}] 内存清理完成，使用 {torch.cuda.memory_allocated() / 1024**3:.2f} GiB 已分配")

def plot_all_models_curves(model_names,dataset_name):
    """
    加载所有模型的 ROC/PR 数据和 metrics，绘制 B: AUROC, C: AUPRC 曲线。
    新增: 标签显示最高性能 (AUROC for B, AUPRC for C)。
    """
    
    dataset_folder = f"{dataset_name}/"
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan', 'magenta', 'yellow']  # 自定义颜色
    
    for i, model_name in enumerate(model_names):
        # 加载 ROC 数据
        roc_path = f"{dataset_folder}{model_name}_roc_data.csv"
        roc_df = pd.read_csv(roc_path)
        fpr, tpr = roc_df['FPR'], roc_df['TPR']
        
        # 加载 metrics 获取 AUROC
        metrics_path = f"{dataset_folder}{model_name}_metrics.csv"
        metrics_df = pd.read_csv(metrics_path)
        auc_value = metrics_df['AUC_ROC'].iloc[0]
        
        # B 面板标签: 模型名 + AUROC
        label_b = f"{model_name} ({auc_value:.3f})"
        ax1.plot(fpr, tpr, color=colors[i % len(colors)], label=label_b, linewidth=2)
        
        # 加载 PR 数据
        pr_path = f"{dataset_folder}{model_name}_pr_data.csv"
        pr_df = pd.read_csv(pr_path)
        precision, recall = pr_df['Precision'], pr_df['Recall']
        
        # C 面板标签: 模型名 + AUPRC
        auprc_value = metrics_df['PRAUC'].iloc[0]
        label_c = f"{model_name} ({auprc_value:.3f})"
        ax2.plot(recall, precision, color=colors[i % len(colors)], label=label_c, linewidth=2)
    
    
    # AUROC 基线
    ax1.plot([0, 1], [0, 1], color='gray', linestyle='--', label='Random (0.500)', linewidth=1.5)
    
    # AUPRC 基线
    ax2.axhline(y=0.5, color='gray', linestyle='--', label=f'Random ({0.5:.3f})', linewidth=1.5)
    
        
    # B: AUROC 面板
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title(f'AUROC curves for all models on IEDB dataset')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # C: AUPRC 面板
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title(f'AUPRC curves for all models on IEDB dataset')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存文件名 (带数据集)
    plot_filename = f"{dataset_name}_all_models_roc_pr.png" if dataset_name else "all_models_roc_pr.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"✅ 曲线图已保存为: {plot_filename}")



if __name__ == '__main__':
    data_file ="/home/dengyang/code/My_code/data/processed_vdjdb.csv"
    print(f"加载数据文件: {data_file}")
    df = load_and_preprocess_data(data_file)

    # 提取数据集名称 (e.g., 'iedb.csv' -> 'IEDB')
    dataset_name = os.path.splitext(os.path.basename(data_file))[0].upper()
    print(f"提取数据集名称: {dataset_name}")


    # --- 方便的模型选择区域 ---      
    # 选择您要运行的模型：
    #model_to_run = 'TEIM'      # pMTnet模型 
    #run_model_comparison(df, model_to_run,dataset_name)

    # 如果要运行所有模型进行对比，可以取消注释以下代码：
    for model_name in ['DeepAntigen','TEIM','MixTCRpred','TPepRet','UniPMT','TCRBagger','ERGO-II','PRIME2.0','pMTnet','UnifyImmun','NetTCR-2.0','ImRex','DLpTCR',  'DeepTCR','TITAN','TEINet','PanPep', 'PISTE']:
            run_model_comparison(df, model_name, dataset_name)
            
            print(f"内存清理完成，使用 {torch.cuda.memory_allocated() / 1024**3:.2f} GiB")

    # 调用（假设你的模型列表）
    model_names = ['ERGO-II','NetTCR-2.0','ImRex','DLpTCR', 'pMTnet','DeepTCR','TITAN','PRIME2.0','TEINet','PanPep','TEIM', 'PISTE','MixTCRpred', 'TPepRet','UniPMT', 'UnifyImmun','TCRBagger','DeepAntigen']  # 你的所有模型
    plot_all_models_curves(model_names, dataset_name)
        
