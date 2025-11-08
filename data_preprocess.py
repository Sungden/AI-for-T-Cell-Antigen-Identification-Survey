# -*- coding: utf-8 -*-
# preprocess_datasets.py
# 这个脚本用于预处理 IEDB, McPAS-TCR, VDJdb 数据集，生成处理后的 CSV 文件。
# 每个数据集单独保存为 processed_[dataset].csv。
# 统一格式：使用 rename 字典，过滤 NaN，trim_tcr，drop_duplicates，保存。
# 假设 required_cols = [col_epitope, col_va, col_ja, col_cdr3a, col_vb, col_jb, col_cdr3b, col_mhc]
# 定义 trim_tcr 函数 (简化版：移除 C 和 F 端，如果是 beta=True)。
# 运行前确保原始文件在同一目录下。

import pandas as pd
import os
import numpy as np


# 定义列名变量 (统一)
col_epitope = 'Epitope'
col_va = 'V_alpha'
col_ja = 'J_alpha'
col_cdr3a = 'CDR3_alpha'
col_vb = 'V_beta'
col_jb = 'J_beta'
col_cdr3b = 'CDR3_beta'
col_mhc = 'MHC'

required_cols = [col_epitope, col_va, col_ja, col_cdr3a, col_vb, col_jb, col_cdr3b, col_mhc]

def trim_tcr(cdr3, is_beta=True):
    """
    简单 trim TCR CDR3：移除 C (N端) 和 F/Y (C端)，beta=True 为 beta 链。
    (自定义函数；可根据需要调整)
    """
    if pd.isna(cdr3) or not isinstance(cdr3, str) or len(cdr3) < 3:
        return ''
    cdr3 = cdr3.strip().upper()
    if is_beta:
        # Beta: 移除 C 和 F/Y
        if cdr3.startswith('C') and cdr3.endswith(('F', 'Y')):
            return cdr3[1:-1]
        elif cdr3.startswith('C'):
            return cdr3[1:]
        elif cdr3.endswith(('F', 'Y')):
            return cdr3[:-1]
    else:
        # Alpha: 类似，但结束可能不同
        if cdr3.startswith('C') and cdr3.endswith(('F', 'Y')):
            return cdr3[1:-1]
        elif cdr3.startswith('C'):
            return cdr3[1:]
        elif cdr3.endswith(('F', 'Y')):
            return cdr3[:-1]
    return cdr3

# IEDB 预处理 (你的原代码)
def process_iedb(path_iedb):
    rename_iedb = {
        'Description': col_epitope,  
        'Calculated V Gene': col_va,
        'Calculated J Gene': col_ja,
        'CDR3 Curated': col_cdr3a,
        'Calculated V Gene.1' : col_vb,
        'Calculated J Gene.1': col_jb,
        'CDR3 Curated.1': col_cdr3b,
        'MHC Allele Names': col_mhc,
        'Name': col_epitope
    }
    df_iedb = pd.read_csv(path_iedb, skiprows=1)
    df_iedb = df_iedb.rename(columns=rename_iedb)
    df_iedb = df_iedb[required_cols]
    df_iedb = df_iedb[~df_iedb[col_epitope].isna()]
    df_iedb = df_iedb[~(df_iedb[col_cdr3a].isna() & df_iedb[col_cdr3b].isna())]
    df_iedb[col_cdr3b] = df_iedb[col_cdr3b].apply(lambda x: trim_tcr(x, True))
    df_iedb[col_cdr3a] = df_iedb[col_cdr3a].apply(lambda x: trim_tcr(x, False))
    df_iedb = df_iedb.drop_duplicates([col_epitope, col_cdr3a, col_cdr3b])
    df_iedb = df_iedb.reset_index(drop=True)
    output_file = 'processed_iedb.csv'
    df_iedb.to_csv(output_file, index=False)
    print(f"✅ IEDB 处理完成，保存到: {output_file} (样本数: {len(df_iedb)})")
    return df_iedb.head()

# McPAS 预处理 (统一格式)
def process_mcpas(path_mcpas):
    rename_mcpas = {
        'CDR3.alpha.aa': col_cdr3a,
        'CDR3.beta.aa': col_cdr3b,
        'TRAV': col_va,
        'TRAJ': col_ja,
        'TRBV': col_vb,
        'TRBJ': col_jb,
        'Epitope.peptide': col_epitope,
        'MHC': col_mhc
    }
    df_mcpas = pd.read_csv(path_mcpas)
    df_mcpas = df_mcpas.rename(columns=rename_mcpas)
    df_mcpas = df_mcpas[required_cols]
    df_mcpas = df_mcpas[~df_mcpas[col_epitope].isna()]
    df_mcpas = df_mcpas[~(df_mcpas[col_cdr3a].isna() & df_mcpas[col_cdr3b].isna())]
    df_mcpas[col_cdr3b] = df_mcpas[col_cdr3b].apply(lambda x: trim_tcr(x, True))
    df_mcpas[col_cdr3a] = df_mcpas[col_cdr3a].apply(lambda x: trim_tcr(x, False))
    df_mcpas = df_mcpas.drop_duplicates([col_epitope, col_cdr3a, col_cdr3b])
    df_mcpas = df_mcpas.reset_index(drop=True)
    output_file = 'processed_mcpas.csv'
    df_mcpas.to_csv(output_file, index=False)
    print(f"✅ McPAS 处理完成，保存到: {output_file} (样本数: {len(df_mcpas)})")
    return df_mcpas.head()

# VDJdb 预处理 (统一格式)
def process_vdjdb(path_vdjdb):
    rename_vdjdb = {
        'cdr3': col_cdr3b,  # 只取 TRB 的 cdr3 作为 beta
        'v.segm': col_vb,
        'j.segm': col_jb,
        'antigen.epitope': col_epitope,
        'mhc.a': col_mhc  # 优先 mhc.a
    }
    df_vdjdb = pd.read_csv(path_vdjdb, sep='\t')
    # 过滤 TRB (beta chain)
    df_vdjdb = df_vdjdb[df_vdjdb['gene'] == 'TRB'].copy()
    df_vdjdb = df_vdjdb.rename(columns=rename_vdjdb)
    # 填充缺失列 (VDJdb 缺少 alpha，设 NaN)
    df_vdjdb[col_cdr3a] = np.nan
    df_vdjdb[col_va] = np.nan
    df_vdjdb[col_ja] = np.nan
    df_vdjdb = df_vdjdb[required_cols]
    df_vdjdb = df_vdjdb[~df_vdjdb[col_epitope].isna()]
    df_vdjdb = df_vdjdb[~df_vdjdb[col_cdr3b].isna()]  # beta 必须有
    df_vdjdb[col_cdr3b] = df_vdjdb[col_cdr3b].apply(lambda x: trim_tcr(x, True))
    df_vdjdb = df_vdjdb.drop_duplicates([col_epitope, col_cdr3a, col_cdr3b])
    df_vdjdb = df_vdjdb.reset_index(drop=True)
    output_file = 'processed_vdjdb.csv'
    df_vdjdb.to_csv(output_file, index=False)
    print(f"✅ VDJdb 处理完成，保存到: {output_file} (样本数: {len(df_vdjdb)})")
    return df_vdjdb.head()

if __name__ == '__main__':
    # 文件路径 (调整为你的路径)
    path_iedb = '/home/dengyang/code/benchmark_TCRprediction/data/raw/iedb.csv'  # 你的 IEDB
    path_mcpas = '/home/dengyang/code/benchmark_TCRprediction/data/raw/McPAS-TCR.csv'
    path_vdjdb = '/home/dengyang/code/benchmark_TCRprediction/data/raw/vdjdb.txt'
    
    # 处理每个数据集
    process_iedb(path_iedb)
    process_mcpas(path_mcpas)
    process_vdjdb(path_vdjdb)
    
    print("\n所有数据集预处理完成！使用 processed_*.csv 作为 main.py 的 data_file。")

