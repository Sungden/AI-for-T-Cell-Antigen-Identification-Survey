# -*- coding: utf-8 -*-

# config.py

import torch

# --- 1. 全局环境配置 ---
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TCR_COLS = ['CDR3_beta', 'V_beta', 'J_beta']
PMHC_COLS = ['Epitope', 'MHC']

# --- 2. 氨基酸常量 ---
ALL_AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'
AMINO_ACID_MAPPING = {aa: i for i, aa in enumerate(ALL_AMINO_ACIDS)}
N_AA = len(ALL_AMINO_ACIDS) # 20

# --- 3. 训练参数 ---
TRAIN_PARAMS = {
    'epochs': 40,
    'batch_size': 64,
    'learning_rate': 0.0002,
    'weight_decay': 1e-5
}