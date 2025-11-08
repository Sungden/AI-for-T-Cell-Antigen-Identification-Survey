import pandas as pd
import numpy as np

# ========== 用户自定义部分 ==========
# 你的训练集路径（IEDB、VDJdb、McPAS-TCR）
train_files = [
    "/home/dengyang/code/My_code/data/processed_iedb.csv",
    "/home/dengyang/code/My_code/data/processed_mcpas.csv",
    "/home/dengyang/code/My_code/data/processed_vdjdb.csv"
]

# ePytope-TCR 基准数据集路径
epy_file = "/home/dengyang/code/My_code/TCR_benchmark_processed_data/processed_data/viral.csv"

# 输出目录
output_prefix = "filtered_epytopeTCR"
# ===================================


# Step 1: 读取训练集并提取 Epitope 集合
def load_train_epitopes(files):
    epitopes = set()
    for f in files:
        df = pd.read_csv(f)
        if 'Epitope' not in df.columns:
            raise ValueError(f"{f} 中找不到 'Epitope' 列")
        epitopes |= set(df['Epitope'].astype(str).str.upper().str.strip())
    return epitopes


# Step 2: 读取 ePytope-TCR 数据
def load_epytope_dataset(epy_file):
    df = pd.read_csv(epy_file)
    required_cols = ['CDR3_beta', 'Epitope', 'Label']  # 可根据数据格式调整
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"{epy_file} 中缺少必须的列: {col}")
    return df


# Step 3: 去交叉
def filter_unseen_epitopes(df_epy, train_epitopes):
    df_epy['Epitope_norm'] = df_epy['Epitope'].astype(str).str.upper().str.strip()
    df_unseen = df_epy[~df_epy['Epitope_norm'].isin(train_epitopes)].copy()
    df_seen = df_epy[df_epy['Epitope_norm'].isin(train_epitopes)].copy()
    return df_unseen, df_seen


# Step 4: （可选）负样本重建
def regenerate_negatives_strict(df_pos, preserve_cols=('V_beta', 'J_beta')):
    """
    在相同 MHC 背景下生成高质量负样本。
    若某个 MHC 下 epitope 唯一，则跳过并提示。
    """
    neg_samples = []
    skipped_mhc = []

    for mhc, group in df_pos.groupby('MHC'):
        epitopes = group['Epitope'].unique()
        if len(epitopes) <= 1:
            skipped_mhc.append(mhc)
            continue

        tcrs = group['CDR3_beta'].values
        shuffled_ep = np.random.permutation(epitopes)
        for i, tcr in enumerate(tcrs):
            new_ep = np.random.choice(epitopes)
            if new_ep == group.iloc[i]['Epitope']:
                # 防止和原epitope相同
                candidates = [e for e in epitopes if e != new_ep]
                if not candidates:
                    continue
                new_ep = np.random.choice(candidates)

            row = group.iloc[i].to_dict()
            row['Epitope'] = new_ep
            row['Label'] = 0
            neg_samples.append(row)

    # 将负样本构成 DataFrame
    if not neg_samples:
        print("⚠️ 未生成任何负样本，可能所有 MHC 都只对应一个 epitope。")
        if skipped_mhc:
            print("跳过的 MHC:", skipped_mhc)
        # 返回一个空 DataFrame 但包含必要列，防止 KeyError
        cols = ['CDR3_beta', 'Epitope', 'MHC', 'Label'] + [c for c in preserve_cols if c in df_pos.columns]
        return pd.DataFrame(columns=cols)

    df_neg = pd.DataFrame(neg_samples)

    # 去重 (TCR, Epitope, MHC)
    pos_pairs = set(zip(df_pos['CDR3_beta'], df_pos['Epitope'], df_pos['MHC']))
    df_neg = df_neg[~df_neg.apply(lambda x: (x['CDR3_beta'], x['Epitope'], x['MHC']) in pos_pairs, axis=1)]

    base_cols = ['CDR3_beta', 'Epitope', 'MHC', 'Label']
    extra_cols = [c for c in preserve_cols if c in df_pos.columns]
    df_neg = df_neg[base_cols + extra_cols]

    print(f"✅ 在 {df_pos['MHC'].nunique()} 个 MHC 背景下生成 {len(df_neg)} 条负样本。")
    if skipped_mhc:
        print(f"⚠️ 跳过 {len(skipped_mhc)} 个 MHC 组（epitope 唯一）：{skipped_mhc}")
    return df_neg



# ========== 主流程 ==========
if __name__ == "__main__":
    print("加载训练集并提取 epitope...")
    train_epitopes = load_train_epitopes(train_files)
    print(f"训练集共包含 {len(train_epitopes)} 个唯一 epitope")

    print("加载 ePytope-TCR 测试数据...")
    df_epy = load_epytope_dataset(epy_file)
    print(f"原始 ePytope-TCR 测试样本数: {len(df_epy)}")

    print("执行去交叉过滤...")
    df_unseen, df_seen = filter_unseen_epitopes(df_epy, train_epitopes)
    print(f"未见表位 (Unseen Epitope) 测试样本数: {len(df_unseen)}")
    print(f"已见表位 (Seen Epitope) 测试样本数: {len(df_seen)}")

    # 为 unseen 集重新生成 MHC 一致的负样本
    df_pos_unseen = df_unseen[df_unseen['Label'] == 1]
    df_neg_unseen = regenerate_negatives_strict(df_pos_unseen, preserve_cols=['V_beta', 'J_beta'])
    df_final_unseen = pd.concat([df_pos_unseen, df_neg_unseen], ignore_index=True)

    # 保存结果
    df_final_unseen.to_csv(f"{output_prefix}_unseen_balanced_strict.csv", index=False)
    print("✅ 去交叉处理完成！")
    print(f"已保存: {output_prefix}_unseen_balanced_strict.csv")
