# -*- coding: utf-8 -*-

# model.py

import torch
import torch.nn as nn
import numpy as np
import pandas as pd 
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, precision_recall_curve, auc
from config import DEVICE, N_AA,SEED 
import torch.nn.functional as F
from typing import Optional
import math, os,csv
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, precision_recall_curve, auc, roc_curve

# --- 1. 模型一：DeepTCR 风格模型 (CNN + FC) ---
# https://github.com/sidhomj/DeepTCR/blob/master/DeepTCR/DeepTCR.py

class GeneEmbedding(nn.Module):
    """
    可训练嵌入层 for V/J beta 和 MHC (one-hot attr)。
    """
    def __init__(self, attr_dim, embed_dim=48):
        super(GeneEmbedding, self).__init__()
        self.embed = nn.Linear(attr_dim, embed_dim)

    def forward(self, x):
        # x: (batch, attr_dim)
        return self.embed(x)

class DeepTCRConvBlock(nn.Module):
    """
    DeepTCR CNN 块：Conv1d + SELU + BN + MaxPool1d。
    """
    def __init__(self, in_channels, out_channels, kernel_size, pool_size):
        super(DeepTCRConvBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2)
        self.bn = nn.BatchNorm1d(out_channels)
        self.selu = nn.SELU()
        self.pool = nn.MaxPool1d(kernel_size=pool_size)

    def forward(self, x):
        # x: (batch, channels, seq_len)
        x = self.conv(x)
        x = self.bn(x)
        x = self.selu(x)
        x = self.pool(x)
        return x

class DeepTCRPredictor(nn.Module):
    """
    DeepTCR 风格模型：序列 CNN + 基因/MHC 嵌入 + 融合 MLP 分类。
    基于官方 DeepTCR 架构，适配 one-hot 输入和属性特征。
    简化：VAE 编码器部分 (conv stack) + 嵌入；无 decoder；焦点在监督预测。
    """
    def __init__(self, max_cdr3_len, max_epitope_len, attr_dim, embed_dim=48, size_of_net='medium'):
        super(DeepTCRPredictor, self).__init__()
        
        self.max_cdr3_len = max_cdr3_len
        self.max_epitope_len = max_epitope_len
        self.attr_dim = attr_dim
        
        if size_of_net == 'small':
            units = [12, 32, 64]
        elif size_of_net == 'medium':
            units = [32, 64, 128]
        elif size_of_net == 'large':
            units = [64, 128, 256]
        else:
            units = size_of_net  # list of units
        
        # CDR3 (beta seq) CNN: multi-layer conv
        self.cdr3_conv1 = DeepTCRConvBlock(N_AA, units[0], kernel_size=5, pool_size=2)
        self.cdr3_conv2 = DeepTCRConvBlock(units[0], units[1], kernel_size=3, pool_size=2)
        self.cdr3_conv3 = DeepTCRConvBlock(units[1], units[2], kernel_size=3, pool_size=2)
        
        # Epitope CNN: similar but shallower
        self.epi_conv1 = DeepTCRConvBlock(N_AA, units[0], kernel_size=3, pool_size=2)
        self.epi_conv2 = DeepTCRConvBlock(units[0], units[1], kernel_size=3, pool_size=2)
        
        # Gene/MHC embedding (V/J beta + MHC in attr_dim)
        self.gene_embed = GeneEmbedding(attr_dim, embed_dim)
        
        # Compute seq_feature_dim
        cdr3_h = max_cdr3_len // (2*2*2)  # three pools of 2
        epi_h = max_epitope_len // (2*2)  # two pools
        self.seq_feature_dim = units[2] * cdr3_h + units[1] * epi_h + embed_dim
        total_input_dim = self.seq_feature_dim
        
        # Classifier MLP
        self.classifier = nn.Sequential(
            nn.Linear(total_input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        seq_len_dim = (self.max_cdr3_len + self.max_epitope_len) * N_AA
        seq_input = x[:, :seq_len_dim]
        attr_input = x[:, seq_len_dim:]
        
        # Reshape sequences
        cdr3_input = seq_input[:, :self.max_cdr3_len * N_AA].view(-1, self.max_cdr3_len, N_AA).transpose(1, 2)  # (batch, N_AA, L_cdr3)
        epi_input = seq_input[:, self.max_cdr3_len * N_AA:].view(-1, self.max_epitope_len, N_AA).transpose(1, 2)  # (batch, N_AA, L_epi)
        
        # CDR3 features
        cdr3_feat1 = self.cdr3_conv1(cdr3_input)
        cdr3_feat2 = self.cdr3_conv2(cdr3_feat1)
        cdr3_feat3 = self.cdr3_conv3(cdr3_feat2)
        cdr3_flat = cdr3_feat3.flatten(start_dim=1)  # (batch, units[2] * h)
        
        # Epitope features
        epi_feat1 = self.epi_conv1(epi_input)
        epi_feat2 = self.epi_conv2(epi_feat1)
        epi_flat = epi_feat2.flatten(start_dim=1)  # (batch, units[1] * h)
        
        # Gene/MHC embed
        gene_embed = self.gene_embed(attr_input)  # (batch, embed_dim)
        
        # Concat
        combined = torch.cat([cdr3_flat, epi_flat, gene_embed], dim=1)
        
        return self.classifier(combined)



# --- 2. 模型二：pMTnet 模型 (独立 CNN Encoder + 融合) ---
# https://github.com/tianshilu/pMTnet/blob/master/pMTnet.py

########################### Atchley's factors#######################
aa_dict_atchley=dict()
with open("/home/dengyang/code/My_code/Atchley_factors.csv",'r') as aa:
    aa_reader=csv.reader(aa)
    next(aa_reader, None)
    for rows in aa_reader:
        aa_name=rows[0]
        aa_factor=rows[1:len(rows)]
        aa_dict_atchley[aa_name]=np.asarray(aa_factor,dtype='float')

# 此顺序应与您在 config.py 中定义的 ALL_AMINO_ACIDS 保持一致
STANDARD_AA_ORDER = 'ACDEFGHIKLMNPQRSTVWY' 
    
# 过滤并排序 Atchley 矩阵的值
sorted_factors = []
    
for aa in STANDARD_AA_ORDER:
  if aa in aa_dict_atchley:
  # 确保每个氨基酸的因子数组都加入列表
    sorted_factors.append(aa_dict_atchley[aa])

# 使用 numpy.stack 将列表中的所有一维数组堆叠成一个二维矩阵
ATCHLEY_MATRIX = np.stack(sorted_factors)

# ----------------------------
def onehot_flat_to_indices(seq_flat: torch.Tensor, max_len: int, add_one: bool = True):
    """
    从 flattened one-hot 恢复 token IDs:
    seq_flat: (batch, max_len * N_AA)
    returns: (batch, max_len) with tokens in {0 (pad), 1..20}
    """
    if seq_flat.dim() != 2:
        raise ValueError("seq_flat must be 2D (batch, max_len * N_AA)")
    resh = seq_flat.view(-1, max_len, N_AA)
    sums = resh.sum(dim=-1)
    argmax = torch.argmax(resh, dim=-1)  # 0..19
    if add_one:
        tokens = torch.where(sums == 0, torch.zeros_like(argmax), argmax + 1)
    else:
        tokens = torch.where(sums == 0, torch.zeros_like(argmax), argmax)
    return tokens.long()


# ----------------------------
# Projection layers (fixed Atchley / BLOSUM)
# ----------------------------
class AtchleyProjection(nn.Module):
    """
    将 one-hot 或 token 映射到 Atchley factors (channels=5).
    支持输入：one-hot flattened (batch, seq_len, N_AA) 或 (batch, seq_len*N_AA) flattened.
    """
    def __init__(self, device=DEVICE):
        super().__init__()
        # register real Atchley matrix to buffer (so it's on same device as model)
        self.register_buffer('atchley', torch.tensor(ATCHLEY_MATRIX, dtype=torch.float32, device=device))

    def forward(self, x):
        """
        x: either (batch, seq_len, N_AA) OR (batch, seq_len*N_AA)
        returns: (batch, 5, seq_len)
        """
        if x.dim() == 2:
            # flattened
            # infer seq_len:
            seq_len = x.size(1) // N_AA
            x = x.view(-1, seq_len, N_AA)
        # x: (b, seq_len, N_AA)
        # find index per position:
        idx = torch.argmax(x, dim=-1)  # (b, seq_len) values 0..19, if all zeros gives 0 (but could be ambiguous)
        # map via atchley: (20,5) -> for each idx pick row
        batch, seq_len = idx.size()
        # idx -> one hot indexing of atchley
        # use gather: expand atchley to (b, seq_len, 5) by indexing
        atch = self.atchley[idx]  # (b, seq_len, 5)
        return atch.transpose(1, 2)  # (b, 5, seq_len)


class TCRAutoencoder_pMT(nn.Module):
    """
    作为 pMTnet 的 TCR 编码器（根据 pMTnet 思路使用 Atchley factors -> conv -> dense）。
    输出固定维度 embed_dim（如 30）。
    """
    def __init__(self, max_cdr3_len: int, embed_dim: int = 30):
        super().__init__()
        self.max_len = max_cdr3_len
        self.atch_proj = AtchleyProjection()
        # encoder convs (channels=5 -> conv)
        self.conv1 = nn.Conv1d(5, 30, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(30)
        self.pool1 = nn.AvgPool1d(kernel_size=4)
        self.conv2 = nn.Conv1d(30, 20, kernel_size=4, padding=2)
        self.bn2 = nn.BatchNorm1d(20)
        self.pool2 = nn.AvgPool1d(kernel_size=4)

        # compute flattened dim dynamically
        # conservative: after two pool4 -> len // 16 (may be zero if len<16) -> handle via adaptive pooling
        self.adaptive = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(20 * 1, embed_dim),
            nn.SELU(),
            nn.Dropout(0.01)
        )
        self.bottleneck = nn.Linear(embed_dim, embed_dim)

    def forward(self, x_flat):
        # x_flat: (b, max_len * N_AA) OR (b, max_len, N_AA)
        if x_flat.dim() == 2:
            seq_len = x_flat.size(1) // N_AA
            x = x_flat.view(-1, seq_len, N_AA)
        else:
            x = x_flat
        atch = self.atch_proj(x)  # (b, 5, seq_len)
        h = F.selu(self.bn1(self.conv1(atch)))
        h = self.pool1(h)
        h = F.selu(self.bn2(self.conv2(h)))
        h = self.pool2(h)
        h = self.adaptive(h)  # (b, 20, 1)
        flat = h.view(h.size(0), -1)  # (b, 20)
        out = self.fc(flat)
        emb = self.bottleneck(out)
        return emb


class pMTnetPredictor(nn.Module):
    """
    pMTnet style model:
    - TCR encoder uses AtchleyProjection + conv encoder
    - PMHC encoder: simple LSTM on one-hot peptide + attribute linear projection for MHC
    - concatenation -> MLP -> sigmoid (returns probability)
    Note: original pMTnet sometimes outputs ranking — here we produce probability (score).
    """
    def __init__(self, max_cdr3_len: int, max_epitope_len: int, attr_dim: int, tcr_embed_dim: int = 30):
        super().__init__()
        self.max_cdr3_len = max_cdr3_len
        self.max_epitope_len = max_epitope_len
        self.attr_dim = attr_dim

        self.tcr_encoder = TCRAutoencoder_pMT(max_cdr3_len, embed_dim=tcr_embed_dim)

        # peptide encoder: simple LSTM on token embedding (we recover tokens from one-hot)
        vocab_size = 21
        self.pep_embed = nn.Embedding(vocab_size, 16, padding_idx=0)
        self.pep_lstm = nn.LSTM(16, 16, num_layers=1, batch_first=True)

        # attribute (MHC) projection (use attr vector as one-hot over alleles)
        self.hla_proj = nn.Linear(attr_dim, 16)

        # combine
        combined_dim = tcr_embed_dim + 16 + 16
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        seq_len_dim = (self.max_cdr3_len + self.max_epitope_len) * N_AA
        seq_flat = x[:, :seq_len_dim]
        attr_input = x[:, seq_len_dim:]

        # TCR embed via Atchley conv encoder
        tcr_flat = seq_flat[:, :self.max_cdr3_len * N_AA]
        tcr_emb = self.tcr_encoder(tcr_flat)  # (b, tcr_embed_dim)

        # peptide -> tokens -> embedding -> LSTM
        pep_flat = seq_flat[:, self.max_cdr3_len * N_AA:]
        pep_tokens = onehot_flat_to_indices(pep_flat, self.max_epitope_len)  # (b, L)
        pep_emb = self.pep_embed(pep_tokens)  # (b, L, emb)
        _, (h_n, _) = self.pep_lstm(pep_emb)
        # h_n shape: (num_layers, b, hidden) -> take last layer
        pep_feat = h_n[-1]

        hla_feat = self.hla_proj(attr_input)

        combined = torch.cat([tcr_emb, pep_feat, hla_feat], dim=1)
        return self.classifier(combined)


# --- 3. 模型三：ERGO-II 模型 (Bi-GRU Encoder + 融合) ---  # 修改: 添加双MLP支持
# https://github.com/IdoSpringer/ERGO-II (Bi-GRU for seq, embeddings for V/J/MHC, separate MLPs)

class ERGOIIPredictor(nn.Module):
    """
    ERGO-II style model: Bi-GRU encoders for CDR3_beta and Epitope; embeddings for attributes; concat + MLP.
    - Modified: Add two MLPs - one basic (no alpha), one extended (with alpha, but since no alpha, use basic; for future).
    """
    def __init__(self, max_cdr3_len, max_epitope_len, attr_dim, hidden_dim=128, embed_dim=64):
        super().__init__()
        self.max_cdr3_len = max_cdr3_len
        self.max_epitope_len = max_epitope_len
        self.attr_dim = attr_dim
        
        self.cdr3_gru = nn.GRU(N_AA, hidden_dim, bidirectional=True, batch_first=True)
        self.epi_gru = nn.GRU(N_AA, hidden_dim, bidirectional=True, batch_first=True)
        self.gene_embed = GeneEmbedding(attr_dim, embed_dim)
        
        self.seq_feature_dim = hidden_dim * 2 * 2  # bi-gru *2 for each seq
        total_dim = self.seq_feature_dim + embed_dim
        
        # Modified: Two MLPs
        self.basic_classifier = nn.Sequential(  # For no alpha
            nn.Linear(total_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        self.extended_classifier = nn.Sequential(  # For with alpha (placeholder, same dim for now)
            nn.Linear(total_dim, 256),  # If alpha, would add alpha dim
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x, has_alpha=False):  # Add has_alpha flag for future
        seq_len_dim = (self.max_cdr3_len + self.max_epitope_len) * N_AA
        seq_input = x[:, :seq_len_dim]
        attr_input = x[:, seq_len_dim:]
        
        cdr3_input = seq_input[:, :self.max_cdr3_len * N_AA].view(-1, self.max_cdr3_len, N_AA)
        epi_input = seq_input[:, self.max_cdr3_len * N_AA:].view(-1, self.max_epitope_len, N_AA)
        
        cdr3_h, _ = self.cdr3_gru(cdr3_input)
        epi_h, _ = self.epi_gru(epi_input)
        
        cdr3_flat = cdr3_h.mean(dim=1)  # or last: [:, -1, :]
        epi_flat = epi_h.mean(dim=1)
        
        gene_embed = self.gene_embed(attr_input)
        
        combined = torch.cat([cdr3_flat, epi_flat, gene_embed], dim=1)
        
        if has_alpha:
            return self.extended_classifier(combined)
        else:
            return self.basic_classifier(combined)



# --- 4. 模型四：NetTCR-2.0 模型 ---
# https://github.com/mnielLab/NetTCR-2.0

class MultiKernelConv(nn.Module):
    """
    NetTCR multi-kernel 1D conv block (parallel kernels) with global pooling.
    """
    def __init__(self, in_channels: int, num_filters: int = 16, kernel_sizes: Optional[list] = None):
        super().__init__()
        if kernel_sizes is None:
            kernel_sizes = [1, 3, 5, 7, 9]
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels, num_filters, kernel_size=k, padding=k // 2)
            for k in kernel_sizes
        ])
        self.pool = nn.AdaptiveMaxPool1d(1)

    def forward(self, x):
        # x: (b, in_channels, seq_len)
        feats = []
        for conv in self.convs:
            y = conv(x)  # (b, num_filters, seq_len)
            y = F.relu(y)
            y = self.pool(y).squeeze(-1)  # (b, num_filters)
            feats.append(y)
        return torch.cat(feats, dim=1)  # (b, num_filters * len(kernels))


class NetTCR2Predictor(nn.Module):
    """
    NetTCR-2.0 style:
    - MultiKernelConv for peptide and TCR, AdaptiveMaxPool (global)
    - Attribute projection -> concat -> dense -> sigmoid
    """
    def __init__(self, max_cdr3_len: int, max_epitope_len: int, attr_dim: int,
                 num_filters: int = 16, kernel_sizes: Optional[list] = None, attr_embed_dim: int = 32):
        super().__init__()
        self.max_cdr3_len = max_cdr3_len
        self.max_epitope_len = max_epitope_len
        self.attr_dim = attr_dim

        self.tcr_cnn = MultiKernelConv(N_AA, num_filters=num_filters, kernel_sizes=kernel_sizes)
        self.pep_cnn = MultiKernelConv(N_AA, num_filters=num_filters, kernel_sizes=kernel_sizes)

        out_dim_each = num_filters * (len(kernel_sizes) if kernel_sizes is not None else 5)
        self.attr_proj = nn.Linear(attr_dim, attr_embed_dim)

        self.classifier = nn.Sequential(
            nn.Linear(out_dim_each * 2 + attr_embed_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        seq_len_dim = (self.max_cdr3_len + self.max_epitope_len) * N_AA
        seq_flat = x[:, :seq_len_dim]
        attr_input = x[:, seq_len_dim:]

        tcr_flat = seq_flat[:, :self.max_cdr3_len * N_AA].view(-1, self.max_cdr3_len, N_AA).transpose(1, 2)
        pep_flat = seq_flat[:, self.max_cdr3_len * N_AA:].view(-1, self.max_epitope_len, N_AA).transpose(1, 2)

        tcr_feat = self.tcr_cnn(tcr_flat)
        pep_feat = self.pep_cnn(pep_flat)
        attr_feat = self.attr_proj(attr_input)

        combined = torch.cat([tcr_feat, pep_feat, attr_feat], dim=1)
        return self.classifier(combined)



# --- 5. 模型五：ImRex 模型  ---
# https://github.com/pmoris/ImRex


ATCHLEY_MATRIX = np.array([
    [-0.591, -1.302, -0.733,  1.570, -0.146],
    [-1.343,  0.465, -0.862,  1.358, -0.255],
    [ 1.050,  0.302, -3.656, -0.259, -3.242],
    [ 1.357, -1.453,  1.477, -0.049, -0.147],
    [-1.006, -0.590,  1.891, -0.397,  0.412],
    [-0.384,  1.652,  1.330,  1.045,  2.064],
    [-0.336, -0.417, -1.673, -1.474, -0.078],
    [ 1.142, -1.547,  2.131, -0.237,  1.021],
    [-1.423,  0.673, -0.734, -0.098, -0.011],
    [ 0.257, -0.456,  1.273,  0.333, -1.628],
    [ 0.407, -1.453,  0.700, -0.259, -0.462],
    [ 0.945,  0.828,  1.299,  0.933,  1.233],
    [ 0.189,  1.628, -0.517, -0.670, -1.474],
    [ 0.931, -0.179, -0.663, -0.242, -0.813],
    [-1.540, -0.389, -0.746, -1.136,  1.512],
    [-0.228,  1.399,  2.213,  0.595, -2.017],
    [-0.032,  0.326,  1.365, -0.421, -0.510],
    [ 1.212, -1.237, -0.895, -1.020,  1.212],
    [-2.128, -0.324, -0.028, -1.264, -0.478],
    [-0.595, -0.114,  0.392, -1.272, -0.647]
], dtype=np.float32)
# fallback BLOSUM62
BLOSUM62 = np.array([
    [ 4,  0, -2, -1, -2,  0, -2, -1, -1, -1, -1, -2, -1, -1, -1,  1,  0,  0, -3, -2],
    [ 0,  9, -3, -4, -2, -3, -3, -1, -3, -1, -1, -3, -3, -3, -3, -1, -1, -1, -2, -2],
    [-2, -3,  6,  2, -3, -1, -1, -3, -1, -4, -3,  1, -1,  0, -2,  0, -1, -3, -4, -3],
    [-1, -4,  2,  5, -3, -2,  0, -3,  1, -4, -3,  0, -1,  2, -1,  0, -1, -3, -3, -2],
    [-2, -2, -3, -3,  6, -3, -1,  0, -3,  0,  0, -3, -4, -3, -3, -2, -2, -1,  1,  3],
    [ 0, -3, -1, -2, -3,  6, -2, -4, -2, -4, -3,  0, -2, -2, -2,  0, -2, -3, -2, -3],
    [-2, -3, -1,  0, -1, -2,  8, -3, -1, -3, -2,  1, -2,  0,  0, -1, -2, -3, -2,  2],
    [-1, -1, -3, -3,  0, -4, -3,  4, -3,  2,  1, -3, -3, -3, -3, -2, -1,  3, -3, -1],
    [-1, -3, -1,  1, -3, -2, -1, -3,  5, -2, -1,  0, -1,  1,  2,  0, -1, -2, -3, -2],
    [-1, -1, -4, -4,  0, -4, -3,  2, -2,  4,  2, -3, -3, -2, -2, -2, -1,  1, -2, -1],
    [-1, -1, -3, -3,  0, -3, -2,  1, -1,  2,  5, -2, -2,  0, -1, -1, -1,  1, -1, -1],
    [-2, -3,  1,  0, -3,  0,  1, -3,  0, -3, -2,  6, -2,  0,  0,  1,  0, -3, -4, -2],
    [-1, -3, -1, -1, -4, -2, -2, -3, -1, -3, -2, -2,  7, -1, -2, -1, -1, -2, -4, -3],
    [-1, -3,  0,  2, -3, -2,  0, -3,  1, -2,  0,  0, -1,  5,  1,  0, -1, -2, -2, -1],
    [-1, -3, -2, -1, -3, -2,  0, -3,  2, -2, -1,  0, -2,  1,  5, -1, -1, -3, -3, -2],
    [ 1, -1,  0,  0, -2,  0, -1, -2,  0, -2, -1,  1, -1,  0, -1,  4,  1, -2, -3, -2],
    [ 0, -1, -1, -1, -2, -2, -2, -1, -1, -1, -1,  0, -1, -1, -1,  1,  5,  0, -2, -2],
    [ 0, -1, -3, -3, -1, -3, -3,  3, -2,  1,  1, -3, -2, -2, -3, -2,  0,  4, -3, -1],
    [-3, -2, -4, -3,  1, -2, -2, -3, -3, -2, -1, -4, -4, -4, -3, -3, -2, -3, 11,  2],
    [-2, -2, -3, -2,  3, -3,  2, -1, -2, -1, -1, -2, -3, -1, -2, -2, -2, -1,  2,  7]
], dtype=np.float32)


# ----------------------------
# Projection layers (fixed Atchley / BLOSUM)
# ----------------------------
class AtchleyProjectionFixed(nn.Module):
    """
    one-hot (flattened or (b, seq_len, N_AA)) -> Atchley factors (5 channels)
    Output: (batch, 5, seq_len)
    """
    def __init__(self):
        super().__init__()
        mat = torch.tensor(ATCHLEY_MATRIX, dtype=torch.float32)
        self.register_buffer("atchley", mat)  # (20,5)

    def forward(self, x):
        # accept flattened one-hot (b, seq_len*N_AA) or (b, seq_len, N_AA)
        if x.dim() == 2:
            seq_len = x.size(1) // N_AA
            x = x.view(-1, seq_len, N_AA)
        mask = x.sum(dim=-1) > 0
        idx = torch.argmax(x, dim=-1)  # (b, seq_len) in 0..19
        out = self.atchley[idx]  # (b, seq_len, 5)
        out = out.permute(0, 2, 1).contiguous()  # (b, 5, seq_len)
        out = out * mask.unsqueeze(1).float()
        return out


class BLOSUMProjectionFixed(nn.Module):
    """
    one-hot (or tokens) -> BLOSUM62 row vectors (channel = 20)
    Output: (batch, 20, seq_len)
    """
    def __init__(self):
        super().__init__()
        mat = torch.tensor(BLOSUM62, dtype=torch.float32)
        self.register_buffer("blosum", mat)  # (20,20)

    def forward(self, x):
        # if flattened one-hot
        if x.dim() == 2:
            seq_len = x.size(1) // N_AA
            x = x.view(-1, seq_len, N_AA)
        if x.dim() == 3:
            mask = x.sum(dim=-1) > 0
            idx = torch.argmax(x, dim=-1)
        else:
            # assume tokens (b, seq_len)
            idx = x
            mask = (idx != 0)
        out = self.blosum[idx]  # (b, seq_len, 20)
        out = out.permute(0, 2, 1).contiguous()  # (b, 20, seq_len)
        out = out * mask.unsqueeze(1).float()
        return out


class ImRexInteractionCNN(nn.Module):
    def __init__(self, in_channels=1, dim_hidden=64, num_blocks=3):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, dim_hidden, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim_hidden),
            nn.ReLU(inplace=True)
        )
        blocks = []
        for _ in range(num_blocks):
            blocks.append(nn.Sequential(
                nn.Conv2d(dim_hidden, dim_hidden, kernel_size=3, padding=1),
                nn.BatchNorm2d(dim_hidden),
                nn.ReLU(inplace=True),
                nn.Conv2d(dim_hidden, dim_hidden, kernel_size=3, padding=1),
                nn.BatchNorm2d(dim_hidden)
            ))
        self.blocks = nn.ModuleList(blocks)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.AdaptiveMaxPool2d((1, 1))

    def forward(self, x):
        x = self.initial(x)
        for block in self.blocks:
            res = block(x)
            x = self.relu(x + res)
        x = self.pool(x).view(x.size(0), -1)
        return x


class ImRexPredictor(nn.Module):
    def __init__(self, max_cdr3_len, max_epitope_len, attr_dim,
                 feat_mode='atchley', inter_dim=32, cnn_hidden=64, num_blocks=3, dropout=0.2):
        super().__init__()
        self.max_cdr3_len = max_cdr3_len
        self.max_epitope_len = max_epitope_len
        self.attr_dim = attr_dim
        self.feat_mode = feat_mode

        # choose feature projection based on feat_mode
        if feat_mode == 'atchley':
            self.feat_proj = AtchleyProjectionFixed()
            c_in = ATCHLEY_MATRIX.shape[1]  # 5
        elif feat_mode == 'blosum':
            self.feat_proj = BLOSUMProjectionFixed()
            c_in = BLOSUM62.shape[1]  # 20
        else:
            self.feat_proj = None
            c_in = None

        # project per-channel to inter_dim
        if self.feat_proj is not None:
            self.proj_tcr = nn.Conv1d(c_in, inter_dim, kernel_size=1)
            self.proj_epi = nn.Conv1d(c_in, inter_dim, kernel_size=1)
        else:
            # fallback: token embedding path
            self.embed_tok = nn.Embedding(21, inter_dim, padding_idx=0)
            self.proj_tcr = None
            self.proj_epi = None

        self.inter_cnn = ImRexInteractionCNN(in_channels=inter_dim, dim_hidden=cnn_hidden, num_blocks=num_blocks)
        self.attr_fc = nn.Sequential(nn.Linear(attr_dim, 64), nn.ReLU(), nn.Dropout(dropout))
        self.classifier = nn.Sequential(nn.Linear(cnn_hidden + 64, 128), nn.ReLU(), nn.Dropout(dropout),
                                        nn.Linear(128, 1), nn.Sigmoid())

    def forward(self, x):
        seq_len_dim = (self.max_cdr3_len + self.max_epitope_len) * N_AA
        seq_flat = x[:, :seq_len_dim]
        attr_input = x[:, seq_len_dim:]

        # get tcr / peptide flat segments
        tcr_flat = seq_flat[:, :self.max_cdr3_len * N_AA]
        pep_flat = seq_flat[:, self.max_cdr3_len * N_AA:]

        if self.feat_proj is not None:
            tcr_feat = self.feat_proj(tcr_flat)  # (b, C_in, L_tcr)
            epi_feat = self.feat_proj(pep_flat)  # (b, C_in, L_epi)
            tcr_proj = self.proj_tcr(tcr_feat)  # (b, inter_dim, L_tcr)
            epi_proj = self.proj_epi(epi_feat)  # (b, inter_dim, L_epi)
        else:
            # token path: recover tokens -> embedding -> transpose
            tcr_tokens = onehot_flat_to_indices(tcr_flat, self.max_cdr3_len)
            epi_tokens = onehot_flat_to_indices(pep_flat, self.max_epitope_len)
            tcr_emb = self.embed_tok(tcr_tokens).transpose(1, 2)
            epi_emb = self.embed_tok(epi_tokens).transpose(1, 2)
            tcr_proj = tcr_emb
            epi_proj = epi_emb

        # interaction map: outer product per channel -> (b, inter_dim, L_tcr, L_epi)
        t = tcr_proj.unsqueeze(-1)  # (b, c, Lt, 1)
        e = epi_proj.unsqueeze(-2)  # (b, c, 1, Le)
        inter_map = t * e  # (b, c, Lt, Le)

        inter_feat = self.inter_cnn(inter_map)
        attr_feat = self.attr_fc(attr_input)
        combined = torch.cat([inter_feat, attr_feat], dim=1)
        return self.classifier(combined)


        
        
# --- 6. 模型六：TEIM 模型 (CNN-LSTM 混合编码) ---
# https://github.com/pengxingang/TEIM

class ResNet(nn.Module):
    """
    TEIM ResNet block: residual connection for Conv2d。
    """
    def __init__(self, cnn):
        super(ResNet, self).__init__()
        self.cnn = cnn

    def forward(self, data):
        tmp_data = self.cnn(data)
        out = tmp_data + data
        return out

class View(nn.Module):
    """
    TEIM View: reshape tensor to specified shape。
    """
    def __init__(self, *shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, input):
        shape = [input.shape[0]] + list(self.shape)
        return input.view(*shape)

class AutoEncoder(nn.Module):
    """
    TEIM AutoEncoder for epitope pretraining (optional, disabled by default)。
    """
    def __init__(self, dim_hid, len_seq, dim_emb=128, vocab_size=21):
        super(AutoEncoder, self).__init__()
        # Simulate embedding (vocab_size=21 for AA + pad=0; dim_emb=128)
        self.embedding_module = nn.Embedding(vocab_size, dim_emb, padding_idx=0)
        self.encoder = nn.Sequential(
            nn.Conv1d(dim_emb, dim_hid, 3, padding=1),
            nn.BatchNorm1d(dim_hid),
            nn.ReLU(),
            nn.Conv1d(dim_hid, dim_hid, 3, padding=1),
            nn.BatchNorm1d(dim_hid),
            nn.ReLU(),
        )
        self.seq2vec = nn.Sequential(
            nn.Flatten(),
            nn.Linear(len_seq * dim_hid, dim_hid),
            nn.ReLU()
        )
        self.vec2seq = nn.Sequential(
            nn.Linear(dim_hid, len_seq * dim_hid),
            nn.ReLU(),
            View(dim_hid, len_seq)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(dim_hid, dim_hid, kernel_size=3, padding=1),
            nn.BatchNorm1d(dim_hid),
            nn.ReLU(),
            nn.ConvTranspose1d(dim_hid, dim_hid, kernel_size=3, padding=1),
            nn.BatchNorm1d(dim_hid),
            nn.ReLU(),
        )
        self.out_layer = nn.Linear(dim_hid, vocab_size)

    def forward(self, inputs, latent_only=False):
        seq_emb = self.embedding_module(inputs)
        seq_enc = self.encoder(seq_emb.transpose(1, 2))
        vec = self.seq2vec(seq_enc)
        seq_repr = self.vec2seq(vec)
        seq_dec = self.decoder(seq_repr)
        out = self.out_layer(seq_dec.transpose(1, 2))
        if latent_only:
            return vec
        else:
            return out, seq_enc, vec

class TEIMPredictor(nn.Module):
    """
    TEIM 整体模型：Embedding + Conv1d seq feats + Interaction Map (cat/add/mul/outer) + Multi-layer CNN (ResNet) + Seq-level prediction。
    基于仓库models.py：inter_type='cat'默认；无AE (use_ae=False)；适配one-hot输入 (recover aa indices +1 for embedding, pad=0 implicit)。
    整合属性：flatten后cat attr_input到classifier。
    默认config: dim_hidden=128, layers_inter=3, inter_type='cat'。
    修复：输出形状为(batch, 1) 以匹配Y (batch, 1)。
    """
    def __init__(self, max_cdr3_len, max_epitope_len, attr_dim, dim_hidden=128, layers_inter=3, inter_type='cat', use_ae=False):
        super(TEIMPredictor, self).__init__()
        
        self.max_cdr3_len = max_cdr3_len
        self.max_epitope_len = max_epitope_len
        self.attr_dim = attr_dim
        self.dim_hidden = dim_hidden
        self.layers_inter = layers_inter
        self.inter_type = inter_type
        self.use_ae = use_ae
        
        # Embedding (vocab=21: 0=pad, 1-20=AA; dim_emb=128)
        vocab_size = 21
        dim_emb = 128
        self.embedding_module = nn.Embedding(vocab_size, dim_emb, padding_idx=0)
        
        # Sequence feature extractors
        self.seq_cdr3 = nn.Sequential(
            nn.Conv1d(dim_emb, dim_hidden, 1),
            nn.BatchNorm1d(dim_hidden),
            nn.ReLU(),
        )
        self.seq_epi = nn.Sequential(
            nn.Conv1d(dim_emb, dim_hidden, 1),
            nn.BatchNorm1d(dim_hidden),
            nn.ReLU(),
        )
        
        # Interaction map combiner
        if inter_type == 'cat':
            self.combine_layer = nn.Conv2d(dim_hidden * 2, dim_hidden, 1, bias=False)
        elif inter_type == 'outer':
            self.combine_layer = nn.Conv2d(dim_hidden * dim_hidden, dim_hidden, 1, bias=False)
        elif inter_type in ['add', 'mul']:
            self.combine_layer = None
        else:
            raise ValueError(f"Unsupported inter_type: {inter_type}")
        
        # Interaction layers (fixed: always include first two, append extras if >2)
        inter_layers_list = [
            nn.Sequential(  # First layer
                ResNet(nn.Conv2d(dim_hidden, dim_hidden, kernel_size=3, padding=1)),
                nn.BatchNorm2d(dim_hidden),
                nn.ReLU(),
            ),
            nn.ModuleList([  # Second layer (structure for potential AE add)
                ResNet(nn.Conv2d(dim_hidden, dim_hidden, kernel_size=3, padding=1)),
                nn.Sequential(
                    nn.BatchNorm2d(dim_hidden),
                    nn.ReLU(),
                ),
            ]),
        ]
        if layers_inter > 2:
            for _ in range(layers_inter - 2):
                inter_layers_list.append(
                    nn.Sequential(
                        ResNet(nn.Conv2d(dim_hidden, dim_hidden, kernel_size=3, padding=1)),
                        nn.BatchNorm2d(dim_hidden),
                        nn.ReLU(),
                    )
                )
        self.inter_layers = nn.ModuleList(inter_layers_list)
        
        # AE (disabled by default)
        if use_ae:
            self.ae_encoder = AutoEncoder(dim_hidden, max_epitope_len)
            self.ae_linear = nn.Linear(dim_hidden, dim_hidden, bias=False)
        else:
            self.ae_encoder = None
        
        # Post-processing components
        self.pool = nn.AdaptiveMaxPool2d(1)
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Sequential(
            nn.Linear(dim_hidden + attr_dim, 1),
            nn.Sigmoid()
        )
        
        # Seq feature dim (for logging)
        self.seq_feature_dim = dim_hidden + attr_dim
        
    def recover_aa_indices(self, onehot_flat, max_len):
        """从flatten one-hot (N_AA=20)恢复aa indices (batch, max_len)，+1 to 1-20 for embedding。"""
        onehot_reshaped = onehot_flat.view(-1, max_len, 20)  # N_AA=20
        aa_indices = torch.argmax(onehot_reshaped, dim=-1) + 1  # 0-19 -> 1-20
        return aa_indices.long()
    
    def forward(self, x):
        # 解包输入
        seq_len_dim = (self.max_cdr3_len + self.max_epitope_len) * 20  # N_AA=20
        seq_input = x[:, :seq_len_dim]
        attr_input = x[:, seq_len_dim:]
        
        # Recover aa indices
        cdr3_aa = self.recover_aa_indices(seq_input[:, :self.max_cdr3_len * 20], self.max_cdr3_len)
        epi_aa = self.recover_aa_indices(seq_input[:, self.max_cdr3_len * 20 :], self.max_epitope_len)
        len_cdr3, len_epi = self.max_cdr3_len, self.max_epitope_len
        
        # Embedding
        cdr3_emb = self.embedding_module(cdr3_aa)  # (batch, len_cdr3, dim_emb)
        epi_emb = self.embedding_module(epi_aa)     # (batch, len_epi, dim_emb)
        
        # Sequence features
        cdr3_feat = self.seq_cdr3(cdr3_emb.transpose(1, 2))  # (batch, dim_hidden, len_cdr3)
        epi_feat = self.seq_epi(epi_emb.transpose(1, 2))      # (batch, dim_hidden, len_epi)
        
        ae_feat = None
        if self.use_ae and self.ae_encoder is not None:
            ae_feat = self.ae_encoder(epi_aa, latent_only=True)  # (batch, dim_hidden)
            ae_feat = self.ae_linear(ae_feat)
        
        # Initial interaction map
        cdr3_feat_mat = cdr3_feat.unsqueeze(3).repeat(1, 1, 1, len_epi)  # (batch, h, len_cdr3, len_epi)
        epi_feat_mat = epi_feat.unsqueeze(2).repeat(1, 1, len_cdr3, 1)    # (batch, h, len_cdr3, len_epi)
        
        if self.inter_type == 'cat':
            inter_feat = torch.cat([cdr3_feat_mat, epi_feat_mat], dim=1)
            inter_map = self.combine_layer(inter_feat)
        elif self.inter_type == 'outer':
            cdr3_feat_mat = cdr3_feat_mat.unsqueeze(2)  # (b, h, 1, L_cdr3, L_epi)
            epi_feat_mat = epi_feat_mat.unsqueeze(1)    # (b, 1, h, L_cdr3, L_epi)
            inter_map = cdr3_feat_mat * epi_feat_mat    # (b, h, h, L_cdr3, L_epi)
            inter_map = self.combine_layer(inter_map.view(inter_map.shape[0], -1, len_cdr3, len_epi))
        elif self.inter_type == 'add':
            inter_map = cdr3_feat_mat + epi_feat_mat
        elif self.inter_type == 'mul':
            inter_map = cdr3_feat_mat * epi_feat_mat
        else:
            raise ValueError(f"Unsupported inter_type: {self.inter_type}")
        
        # Interaction layers processing
        for i in range(self.layers_inter):
            layer = self.inter_layers[i]
            if isinstance(layer, nn.ModuleList):
                inter_map = layer[0](inter_map)
                if i == 1 and self.use_ae and ae_feat is not None:
                    vec = ae_feat.unsqueeze(2).unsqueeze(3)  # (b, h, 1, 1) for broadcast
                    inter_map = inter_map + vec
                inter_map = layer[1](inter_map)
            else:
                inter_map = layer(inter_map)
        
        # Seq-level prediction (pool + flatten + cat attr + classify)
        pooled = self.pool(inter_map)  # (batch, h, 1, 1)
        flattened = pooled.flatten(1)  # (batch, h)
        combined = torch.cat([flattened, attr_input], dim=1)  # (batch, h + attr_dim)
        dropout_out = self.dropout(combined)
        out = self.classifier(dropout_out)  # (batch, 1)
        
        return out  # (batch, 1) to match Y_batch
        


# --- 7. 模型七：MixTCRpred 模型 (双 Bi-LSTM 编码器) ---
# https://github.com/GfellerLab/MixTCRpred


class PositionWiseEmbedding(nn.Module):
    """
    MixTCRpred 位置嵌入：learned positional embedding for sequences。
    """
    def __init__(self, vocab_size, embedding_dim, max_len):
        super(PositionWiseEmbedding, self).__init__()
        self.embedding_dim = embedding_dim
        self.max_len = max_len
        self.pos_embedding = nn.Embedding(self.max_len, embedding_dim)

    def forward(self, x):
        batch_size = x.shape[0]
        inputs_len = x.shape[1]
        pos = torch.arange(0, inputs_len).unsqueeze(0).repeat(batch_size, 1).to(x.device)
        pos_embedding = self.pos_embedding(pos)
        return pos_embedding

def scaled_dot_product(q, k, v, mask=None):
    d_k = q.size()[-1]
    attn_logits = torch.matmul(q, k.transpose(-2, -1))
    attn_logits = attn_logits / math.sqrt(d_k)
    if mask is not None:
        attn_logits = attn_logits.masked_fill(mask == 0, -9e15)
    attention = F.softmax(attn_logits, dim=-1)
    values = torch.matmul(attention, v)
    return values, attention

class MultiheadAttention(nn.Module):
    """
    MixTCRpred Multihead Attention block。
    """
    def __init__(self, input_dim, embed_dim, num_heads):
        super(MultiheadAttention, self).__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.qkv_proj = nn.Linear(input_dim, 3 * embed_dim)
        self.o_proj = nn.Linear(embed_dim, embed_dim)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        self.qkv_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)

    def forward(self, x, mask=None, return_attention=False):
        batch_size, seq_length, embed_dim = x.size()
        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(batch_size, seq_length, self.num_heads, 3 * self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3)
        q, k, v = qkv.chunk(3, dim=-1)
        values, attention = scaled_dot_product(q, k, v, mask=mask)
        values = values.permute(0, 2, 1, 3)
        values = values.reshape(batch_size, seq_length, embed_dim)
        o = self.o_proj(values)
        if return_attention:
            return o, attention
        else:
            return o

class EncoderBlock(nn.Module):
    """
    MixTCRpred Encoder Block: MHAttention + FFN + residuals。
    """
    def __init__(self, input_dim, num_heads, dim_feedforward, dropout=0.0):
        super(EncoderBlock, self).__init__()
        self.self_attn = MultiheadAttention(input_dim, input_dim, num_heads)
        self.linear_net = nn.Sequential(
            nn.Linear(input_dim, dim_feedforward),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True),
            nn.Linear(dim_feedforward, input_dim)
        )
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_out = self.self_attn(x, mask=mask)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)
        linear_out = self.linear_net(x)
        x = x + self.dropout(linear_out)
        x = self.norm2(x)
        return x

class TransformerEncoder(nn.Module):
    """
    MixTCRpred Transformer Encoder: stack of EncoderBlocks。
    """
    def __init__(self, num_layers, input_dim, dim_feedforward, num_heads, dropout):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([
            EncoderBlock(input_dim, num_heads, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x, mask=None):
        for l in self.layers:
            x = l(x, mask=mask)
        return x

class MixTCRpredPredictor(nn.Module):
    """
    MixTCRpred 整体模型：Token + Pos embedding + Transformer Encoder for Epitope/CDR3 + cat attr + MLP分类。
    基于仓库models.py：简化到TRB CDR3 + Epitope (no TRA/CDR1/2)；learned pos embedding；共享Transformer。
    适配框架：one-hot flatten输入，recover tokens (1-20 for AA, 0=pad)；mask for pad；attr cat after flatten。
    默认参数：embedding_dim=128, hidden_dim=512, num_heads=8, num_layers=6, dropout=0.1。
    """
    def __init__(self, max_cdr3_len, max_epitope_len, attr_dim, embedding_dim=128, hidden_dim=512, num_heads=8, num_layers=6, dropout=0.1):
        super(MixTCRpredPredictor, self).__init__()
        
        self.max_cdr3_len = max_cdr3_len
        self.max_epitope_len = max_epitope_len
        self.attr_dim = attr_dim
        self.vocab_size = 21  # 20 AA + pad=0
        self.embedding_dim = embedding_dim
        self.padding_idx = 0
        
        # Token embedding
        self.embedding_tok = nn.Embedding(self.vocab_size, embedding_dim, padding_idx=self.padding_idx)
        
        # Positional embeddings
        self.embedding_pos_epi = PositionWiseEmbedding(self.vocab_size, embedding_dim, self.max_epitope_len)
        self.embedding_pos_cdr3 = PositionWiseEmbedding(self.vocab_size, embedding_dim, self.max_cdr3_len)
        
        # Scale for token emb
        self.scale = math.sqrt(embedding_dim)
        
        # Shared Transformer encoder
        self.transformer_encoder = TransformerEncoder(
            num_layers=num_layers,
            input_dim=embedding_dim,
            dim_feedforward=hidden_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Output dim: (max_epi + max_cdr3) * embedding_dim + attr_dim
        seq_out_dim = (self.max_epitope_len + self.max_cdr3_len) * embedding_dim
        self.output_net = nn.Sequential(
            nn.Linear(seq_out_dim + attr_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embedding_dim, 1),
            nn.Sigmoid()
        )
        
        self.seq_feature_dim = seq_out_dim + attr_dim
        
    def recover_aa_indices(self, onehot_flat, max_len):
        """从one-hot恢复token IDs (batch, max_len)：argmax +1 (0-19 -> 1-20, pad=0 if all zero)。"""
        onehot_reshaped = onehot_flat.view(-1, max_len, 20)  # N_AA=20
        aa_indices = torch.argmax(onehot_reshaped, dim=-1)
        # If all zero (pad), set to 0; else +1
        aa_indices = torch.where(onehot_reshaped.sum(dim=-1) == 0, torch.zeros_like(aa_indices), aa_indices + 1)
        return aa_indices.long()
    
    def forward(self, x):
        # 解包输入
        seq_len_dim = (self.max_cdr3_len + self.max_epitope_len) * 20  # N_AA=20
        seq_input = x[:, :seq_len_dim]
        attr_input = x[:, seq_len_dim:]
        
        # Recover tokens
        cdr3_tokens = self.recover_aa_indices(seq_input[:, :self.max_cdr3_len * 20], self.max_cdr3_len)
        epi_tokens = self.recover_aa_indices(seq_input[:, self.max_cdr3_len * 20 :], self.max_epitope_len)
        
        # Masks (pad != 0)
        mask_cdr3 = (cdr3_tokens != self.padding_idx).unsqueeze(1).unsqueeze(2)
        mask_epi = (epi_tokens != self.padding_idx).unsqueeze(1).unsqueeze(2)
        
        # CDR3: embed + pos + transformer
        cdr3_emb = self.embedding_tok(cdr3_tokens) * self.scale
        cdr3_pos = self.embedding_pos_cdr3(cdr3_tokens)
        cdr3 = cdr3_emb + cdr3_pos
        cdr3_out = self.transformer_encoder(cdr3, mask=mask_cdr3)
        
        # Epitope: embed + pos + transformer
        epi_emb = self.embedding_tok(epi_tokens) * self.scale
        epi_pos = self.embedding_pos_epi(epi_tokens)
        epi = epi_emb + epi_pos
        epi_out = self.transformer_encoder(epi, mask=mask_epi)
        
        # Concat seq outputs
        seq_out = torch.cat([epi_out, cdr3_out], dim=1)
        seq_flat = seq_out.flatten(start_dim=1)  # (batch, (max_epi + max_cdr3) * emb_dim)
        
        # Cat attr + classify
        combined = torch.cat([seq_flat, attr_input], dim=1)
        return self.output_net(combined)


# --- 8. 模型八：UnifyImmun 模型 (Self/Cross-Attention) ---
# https://github.com/hliulab/UnifyImmun

class UnifyImmunPredictor(nn.Module):
    """
    UnifyImmun 模型：自注意力机制
    """
    def __init__(self, max_cdr3_len, max_epitope_len, attr_dim, embed_dim=48):
        super(UnifyImmunPredictor, self).__init__()
        
        self.max_cdr3_len = max_cdr3_len
        self.max_epitope_len = max_epitope_len
        self.attr_dim = attr_dim

        # TCR嵌入层
        self.tcr_embedding = nn.Embedding(N_AA, embed_dim)
        
        # Epitope嵌入层
        self.epitope_embedding = nn.Embedding(N_AA, embed_dim)

        # 自注意力机制
        self.self_attention = nn.MultiheadAttention(embed_dim, num_heads=8, dropout=0.1)
        
        # 属性嵌入
        self.attr_embedding = nn.Linear(attr_dim, embed_dim)

        # 分类器
        self.fc = nn.Sequential(
            nn.Linear(embed_dim * 2 + embed_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        seq_len_dim = (self.max_cdr3_len + self.max_epitope_len) * N_AA
        seq_input = x[:, :seq_len_dim]
        attr_input = x[:, seq_len_dim:]
        
        # 确保输入是整数索引
        tcr_seq = seq_input[:, :self.max_cdr3_len * N_AA].view(-1, self.max_cdr3_len, N_AA)
        tcr_seq = tcr_seq.argmax(dim=-1)  # 获取最大值的索引
        
        epi_seq = seq_input[:, self.max_cdr3_len * N_AA:].view(-1, self.max_epitope_len, N_AA)
        epi_seq = epi_seq.argmax(dim=-1)  # 获取最大值的索引
        
        # 嵌入层
        tcr_emb = self.tcr_embedding(tcr_seq)  # (batch, max_cdr3_len, embed_dim)
        epi_emb = self.epitope_embedding(epi_seq)  # (batch, max_epitope_len, embed_dim)
        
        # 自注意力机制
        tcr_emb = tcr_emb.transpose(0, 1)  # (max_cdr3_len, batch, embed_dim)
        epi_emb = epi_emb.transpose(0, 1)  # (max_epitope_len, batch, embed_dim)
        
        tcr_emb, _ = self.self_attention(tcr_emb, tcr_emb, tcr_emb)
        epi_emb, _ = self.self_attention(epi_emb, epi_emb, epi_emb)
        
        # 属性嵌入
        attr_emb = self.attr_embedding(attr_input)  # (batch, embed_dim)

        # 拼接特征
        combined = torch.cat([tcr_emb.mean(0), epi_emb.mean(0), attr_emb], dim=1)
        
        return self.fc(combined)

                


# --- 9. 模型九：UniPMT 模型 (双 Self-Attention 编码) ---
# https://github.com/ethanmock/UniPMT

class UniPMTPredictor(nn.Module):
    """
    UniPMT 模型：序列编码 (CNN for Epitope/TCR) + MHC Linear + PM/MT MLP残差块 + 元素-wise乘法融合 + 分类。
    基于官方 MolGNN 架构，适配 one-hot 输入和属性特征。
    简化：忽略 GNN (无图数据，使用直接嵌入) 和预加载 npy；硬编码 hidden_size=128, emb_size=64。
    焦点在 pmt_pred: pm_learn (p+m cat + residual MLPs) * mt_learn (m+t cat + MLP) + cross MLP。
    修复：统一 pm_learn 内部维度为 hidden_size=128 (cat后 linear_pm ->128, parts in 128, final_proj ->64)；
          act=LayerNorm(128)；mt_model 输入128->128->64；cross 64->128->1。
    """
    def __init__(self, max_cdr3_len, max_epitope_len, attr_dim, hidden_size=128, emb_size=64):
        super(UniPMTPredictor, self).__init__()
        
        self.max_cdr3_len = max_cdr3_len
        self.max_epitope_len = max_epitope_len
        self.attr_dim = attr_dim
        self.hidden_size = hidden_size
        self.emb_size = emb_size  # config.out_emb_size 等价
        
        # 1. 序列编码器 (Peptide/Epitope 和 TCR/CDR3 使用简单 CNN)
        self.p_encoder = nn.Sequential(  # Epitope CNN
            nn.Conv1d(N_AA, hidden_size, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),  # 全局平均池化到 (batch, hidden_size)
            nn.Flatten()
        )
        self.t_encoder = nn.Sequential(  # TCR/CDR3 CNN
            nn.Conv1d(N_AA, hidden_size, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )
        
        # 2. 投影层：从 CNN 输出 (hidden_size) 到 emb_size
        self.p_proj = nn.Linear(hidden_size, emb_size)
        self.t_proj = nn.Linear(hidden_size, emb_size)
        
        # 3. MHC 嵌入 (从 attr_dim Linear 到 emb_size；假设 attr 包括 MHC one-hot)
        self.m_emb_model = nn.Sequential(
            nn.Linear(attr_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, emb_size)
        )
        
        # 4. PM 学习块 (residual MLPs, 模拟 pm_part1-4 + acts)
        self.pm_part1 = self.pm_block(hidden_size, hidden_size)  # 统一 hidden_size
        self.pm_part2 = self.pm_block(hidden_size, hidden_size)
        self.pm_part3 = self.pm_block(hidden_size, hidden_size)
        self.pm_part4 = self.pm_block(hidden_size, hidden_size)
        self.act = nn.LayerNorm(hidden_size)  # LayerNorm(hidden_size=128)
        self.linear_pm = nn.Linear(emb_size * 2, hidden_size)  # cat(128) -> hidden(128)
        self.pm_final_proj = nn.Linear(hidden_size, emb_size)  # 最终 -> emb(64)
        
        # 5. MT 学习块 (简单 MLP)
        self.mt_model = nn.Sequential(
            nn.Linear(emb_size * 2, hidden_size),  # cat(128) -> hidden(128)
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, emb_size)  # -> emb(64)
        )
        
        # 6. Cross 融合 (element-wise * + MLP)
        self.cross_model = nn.Sequential(
            nn.Linear(emb_size, hidden_size),  # 64 -> 128
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        
        # 7. 维度 (pm_out 和 mt_out 均为 emb_size)
        self.seq_feature_dim = emb_size * 2  # 但实际融合后不直接用
        # total_input_dim 不适用，因为无最终 cat；打印用 emb_size
        
    def pm_block(self, in_size, out_size):
        """残差块：Linear + LeakyReLU + Dropout + Linear"""
        return nn.Sequential(
            nn.Linear(in_size, out_size),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(out_size, out_size),
        )
    
    def pm_learn(self, p_emb, m_emb):
        """PM 残差学习：多层残差 + LayerNorm (全 hidden_size=128)"""
        x = torch.cat((p_emb, m_emb), dim=1)  # [batch, 128]
        res = self.linear_pm(x)  # [batch, 128]
        res = self.act(res)
        out = self.pm_part1(res)  # [batch, 128]
        res = self.act(out + res)
        out = self.pm_part2(res)
        res = self.act(out + res)
        out = self.pm_part3(res)
        res = self.act(out + res)
        out = self.pm_part4(res)
        res = self.act(out + res)
        return self.pm_final_proj(res)  # [batch, 64]
    
    def mt_learn(self, m_emb, t_emb):
        """MT 简单 MLP"""
        x = torch.cat((m_emb, t_emb), dim=1)  # [batch, 128]
        out = self.mt_model(x)  # [batch, 64]
        return out
    
    def pmt_cross(self, pm_out, mt_out):
        """元素-wise 乘法 + MLP + sigmoid"""
        inp = pm_out * mt_out  # [batch, 64]
        out = self.cross_model(inp)  # [batch, 1]
        out = torch.sigmoid(out)
        return out
    
    def forward(self, x):
        # 1. 解包输入张量
        seq_len_dim = (self.max_cdr3_len + self.max_epitope_len) * N_AA
        seq_input = x[:, :seq_len_dim]
        attr_input = x[:, seq_len_dim:]  # (batch, attr_dim)，包括 MHC
        
        # 2. 序列重塑和编码
        # Epitope (p)
        p_seq_flat = seq_input[:, :self.max_epitope_len * N_AA]
        p_reshaped = p_seq_flat.view(-1, self.max_epitope_len, N_AA).transpose(1, 2)  # (batch, N_AA, L_p)
        p_hidden = self.p_encoder(p_reshaped)  # (batch, hidden_size=128)
        p_emb = self.p_proj(p_hidden)  # (batch, emb_size=64)
        
        # TCR (t)
        t_seq_flat = seq_input[:, self.max_epitope_len * N_AA:]
        t_reshaped = t_seq_flat.view(-1, self.max_cdr3_len, N_AA).transpose(1, 2)  # (batch, N_AA, L_t)
        t_hidden = self.t_encoder(t_reshaped)  # (batch, hidden_size=128)
        t_emb = self.t_proj(t_hidden)  # (batch, emb_size=64)
        
        # MHC (m)
        m_emb = self.m_emb_model(attr_input)  # (batch, emb_size=64)
        
        # 3. PM 和 MT 学习
        pm_out = self.pm_learn(p_emb, m_emb)  # (batch, 64)
        mt_out = self.mt_learn(m_emb, t_emb)  # (batch, 64)
        
        # 4. Cross 融合得到预测
        output = self.pmt_cross(pm_out, mt_out)  # (batch, 1)
        
        return output



# --- 10. 模型十：deepAntigen 模型 ---
# https://github.com/JiangBioLab/deepAntigen

class GCNLayer(nn.Module):
    """
    Simplified GCN layer for residue graph: message passing with LeakyReLU.
    Nodes: residues (AA positions), features: one-hot (20-dim), edges: sequential adj.
    """
    def __init__(self, in_dim, hidden_dim):
        super(GCNLayer, self).__init__()
        self.gather = nn.Linear(in_dim + 11, hidden_dim)  # Simulate edge feat (11-dim dummy)
        self.update = nn.Linear(hidden_dim * 2, hidden_dim)
        self.leaky_relu = nn.LeakyReLU(0.1)

    def forward(self, x, adj):
        # x: (batch, nodes, feat), adj: (batch, nodes, nodes)
        # Message: sum over neighbors
        msg = torch.bmm(adj, x)  # Aggregate neighbors
        # Dummy edge feat: all 0 for simplicity
        batch, nodes, _ = msg.shape
        edge_feat = torch.zeros(batch, nodes, 11, device=x.device)
        gathered = self.leaky_relu(self.gather(torch.cat([msg, edge_feat], dim=-1)))
        # Update: cat self + gathered
        updated = self.leaky_relu(self.update(torch.cat([x, gathered], dim=-1)))
        return updated

class SuperNodeAttention(nn.Module):
    """
    Super node communication: multi-head attention between atoms/residues and super node.
    Simplified to single head for residue level.
    """
    def __init__(self, hidden_dim, num_heads=1):
        super(SuperNodeAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.gate = nn.Sigmoid()

    def forward(self, nodes, super_node):
        # nodes: (batch, nodes, h), super: (batch, h)
        batch, n_nodes, h = nodes.shape
        super_exp = super_node.unsqueeze(1).repeat(1, n_nodes, 1)  # (b, n, h)
        
        Q = self.q_proj(nodes).view(batch, n_nodes, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(super_exp).view(batch, n_nodes, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(super_exp).view(batch, n_nodes, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn = F.softmax(torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim), dim=-1)
        out = torch.matmul(attn, V).transpose(1, 2).contiguous().view(batch, n_nodes, h)
        
        # Gate
        gate = self.gate(out)
        nodes = nodes + gate * (super_exp - nodes)  # Adaptive update
        
        # Update super: average nodes
        super_new = nodes.mean(dim=1)
        return nodes, super_new

class DeepAntigenPredictor(nn.Module):
    """
    DeepAntigen 整体模型：序列到图 (residue nodes) + GCN layers + super node attn + top-k pooling + interaction map (attn fusion) + MLP分类.
    基于论文：简化原子级到residue级 (nodes=positions, feat=one-hot AA); edges=sequential adj; 5 GCN layers, h=128; k=20 top-k; multi-head attn for fusion.
    适配框架：one-hot flatten输入，recover one-hot feat (pad=0); attr linear to h cat to pooled feat; sigmoid output.
    
    重要提示：此实现是原始3D结构GNN的极大简化版，改为在1D序列（顺序邻接矩阵）上运行GNN，
    保留了GNN形式但改变了输入内涵，这是对于公平基准测试和统一输入流所必要的妥协。
    """
    def __init__(self, max_cdr3_len, max_epitope_len, attr_dim, hidden_dim=128, gcn_layers=5, top_k=20, num_heads=4):
        super(DeepAntigenPredictor, self).__init__()
        
        self.max_cdr3_len = max_cdr3_len
        self.max_epitope_len = max_epitope_len
        self.attr_dim = attr_dim
        self.hidden_dim = hidden_dim
        self.gcn_layers = gcn_layers
        self.top_k = top_k
        self.num_heads = num_heads
        
        # Initial proj: one-hot (20) to h
        self.node_proj = nn.Linear(20, hidden_dim)
        
        # GCN layers
        self.gcn_layers = nn.ModuleList([
            GCNLayer(hidden_dim, hidden_dim) for _ in range(gcn_layers)
        ])
        
        # Super node attn
        self.super_attn = SuperNodeAttention(hidden_dim, num_heads=1)
        
        # Score for top-k
        self.score_proj = nn.Linear(hidden_dim, 1)
        
        # Interaction attn for fusion (multi-head)
        self.fusion_q = nn.Linear(hidden_dim, hidden_dim)
        self.fusion_kv = nn.Linear(hidden_dim, hidden_dim)
        
        # Attr embed
        self.attr_embed = nn.Linear(attr_dim, hidden_dim)
        
        # Classifier
        pooled_dim = top_k * hidden_dim * 2  # tcr + epi top-k cat
        self.classifier = nn.Sequential(
            nn.Linear(pooled_dim + hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
        self.seq_feature_dim = pooled_dim + hidden_dim
        
    def build_adj(self, max_len):
        """Sequential adj matrix: self + neighbors (band=1)."""
        adj = torch.eye(max_len).unsqueeze(0).repeat(self.batch_size, 1, 1)  # Self-loop (b, max_len, max_len)
        band = 1
        for i in range(1, band + 1):
            adj[:, i:, :-i] += torch.eye(max_len - i).unsqueeze(0).repeat(self.batch_size, 1, 1)
            adj[:, :-i, i:] += torch.eye(max_len - i).unsqueeze(0).repeat(self.batch_size, 1, 1)
        return adj.to(self.device)
    
    def recover_aa_indices(self, onehot_flat, max_len):
        onehot_reshaped = onehot_flat.view(-1, max_len, 20)  # (b, max_len, 20)
        return onehot_reshaped  # Return one-hot directly as feat
    
    def forward(self, x):
        self.batch_size = x.size(0)
        self.device = x.device
        seq_len_dim = (self.max_cdr3_len + self.max_epitope_len) * 20
        seq_input = x[:, :seq_len_dim]
        attr_input = x[:, seq_len_dim:]
        
        # Recover one-hot feats
        tcr_feat_init = self.recover_aa_indices(seq_input[:, :self.max_cdr3_len * 20], self.max_cdr3_len)
        epi_feat_init = self.recover_aa_indices(seq_input[:, self.max_cdr3_len * 20 :], self.max_epitope_len)
        
        # Proj to hidden
        tcr_nodes = self.node_proj(tcr_feat_init)  # (b, max_tcr, h)
        epi_nodes = self.node_proj(epi_feat_init)  # (b, max_epi, h)
        
        # Build adj (sequential)
        adj_tcr = self.build_adj(tcr_nodes.size(1))
        adj_epi = self.build_adj(epi_nodes.size(1))
        
        # GCN + super attn loop
        for layer in self.gcn_layers:
            tcr_nodes = layer(tcr_nodes, adj_tcr)
            super_tcr = tcr_nodes.mean(dim=1)  # Init super
            tcr_nodes, super_tcr = self.super_attn(tcr_nodes, super_tcr)
            
            epi_nodes = layer(epi_nodes, adj_epi)
            super_epi = epi_nodes.mean(dim=1)
            epi_nodes, super_epi = self.super_attn(epi_nodes, super_epi)
        
        # Top-k pooling
        tcr_scores = self.score_proj(tcr_nodes).squeeze(-1)  # (b, max_tcr)
        top_tcr_idx = torch.topk(tcr_scores, self.top_k, dim=1).indices  # (b, k)
        tcr_pooled = torch.gather(tcr_nodes, 1, top_tcr_idx.unsqueeze(-1).repeat(1, 1, self.hidden_dim))  # (b, k, h)
        
        epi_scores = self.score_proj(epi_nodes).squeeze(-1)
        top_epi_idx = torch.topk(epi_scores, self.top_k, dim=1).indices
        epi_pooled = torch.gather(epi_nodes, 1, top_epi_idx.unsqueeze(-1).repeat(1, 1, self.hidden_dim))
        
        # Fusion: simple cat pooled (paper has attn map, but simplify to cat for framework)
        pooled_cat = torch.cat([tcr_pooled, epi_pooled], dim=1)  # (b, 2k, h)
        pooled_flat = pooled_cat.flatten(1)  # (b, 2k*h)
        
        # Attr embed
        attr_feat = self.attr_embed(attr_input)  # (b, h)
        
        # Combined
        combined = torch.cat([pooled_flat, attr_feat], dim=1)
        
        # Classify
        return self.classifier(combined)


# --- 11. 模型十一：PanPep_TCR 模型 (Meta Learning Style: Atchley + Joint Matrix + Self-Attn + CNN) ---
# https://github.com/bm2-lab/PanPep (code not public; based on Nature MI 2023 paper: https://www.nature.com/articles/s42256-023-00619-3)


class AtchleyEncoding:
    """
    PanPep Atchley factor encoding: 20 AA to 5D biochemical vectors (标准值 from Atchley et al.).
    """
    def __init__(self):
        # 标准 Atchley factors (顺序: ACDEFGHIKLMNPQRSTVWY)
        self.factors = torch.tensor([
            [-0.591, -1.302, -0.733,  1.570, -0.146],  # A
            [-1.343,  0.465, -0.862, -1.020, -0.255],  # C
            [ 1.050,  0.302, -3.656, -0.259, -3.242],  # D
            [ 1.357, -1.453,  1.477,  0.113, -0.837],  # E
            [-1.006, -0.590,  1.891, -0.397,  0.412],  # F
            [-0.384,  1.652,  1.330,  1.045,  2.064],  # G
            [ 0.336, -0.417, -1.673, -1.474, -0.078],  # H
            [-1.239, -0.547,  2.131,  0.393,  0.816],  # I
            [ 1.831, -0.561,  0.533, -0.277,  1.648],  # K
            [-1.019, -0.987, -1.505,  1.266, -0.912],  # L
            [-0.663, -1.524,  2.219, -1.005,  1.212],  # M
            [ 0.945,  0.828,  1.299, -0.169,  0.933],  # N
            [ 0.189,  2.081, -1.628,  0.421, -1.392],  # P
            [ 0.931, -0.179, -3.005, -0.503, -1.853],  # Q
            [ 1.538, -0.055,  1.502,  0.440,  2.897],  # R
            [-0.228,  1.399, -4.760,  0.670, -2.647],  # S
            [-0.032,  0.326,  2.213,  0.908,  1.313],  # T
            [-1.337, -0.279, -0.544,  1.242, -1.262],  # V
            [-0.595,  0.009,  0.672, -2.128, -0.184],  # W
            [ 0.260,  0.830,  3.097, -0.838,  1.512]   # Y
        ], dtype=torch.float32)  # (20, 5)

    def encode(self, aa_indices):
        batch, seq_len = aa_indices.shape
        device = aa_indices.device
        encoded = torch.zeros(batch, seq_len, 5, dtype=torch.float32, device=device)
        valid = (aa_indices >= 0) & (aa_indices < 20)
        encoded[valid] = self.factors[aa_indices[valid]].to(device)
        return encoded

class PositionalEncoding_1(nn.Module):
    """
    Standard sinusoidal positional encoding, fixed for odd d_model (避免广播错误)。
    """
    def __init__(self, d_model, max_len=40):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        # 修正奇数 d_model：sin 用完整 even indices (长度 = d//2 +1)，cos 用 odd (d//2)
        num_even = d_model // 2 + d_model % 2
        num_odd = d_model // 2
        pe[:, 0::2] = torch.sin(position * div_term[:num_even])
        pe[:, 1::2] = torch.cos(position * div_term[:num_odd])
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class SelfAttention(nn.Module):
    """
    Single-head self-attention on joint matrix.
    """
    def __init__(self, d_model=5):
        super().__init__()
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)
        attn = F.softmax(Q @ K.transpose(-2, -1) / math.sqrt(x.size(-1)), dim=-1)
        out = attn @ V
        return self.out_proj(out)

class PanPepTCRPredictor(nn.Module):
    """
    PanPep_TCR 模型：Atchley 编码 + 联合矩阵 + Pos Enc + Self-Attn + Linear ReLU + CNN + BN + MaxPool + Flatten + Attr Cat + Classifier。
    固定长度：Epitope=15, TCR=25, Joint=40×5。
    """
    def __init__(self, max_cdr3_len, max_epitope_len, attr_dim, d_model=5):
        super().__init__()
        
        self.max_cdr3_len_input = max_cdr3_len
        self.max_cdr3_len = 25
        self.max_epitope_len_input = max_epitope_len
        self.max_epitope_len = 15
        self.joint_len = 40
        self.d_model = d_model
        self.attr_dim = attr_dim
        self.seq_feature_dim = 608 + d_model  # Post-pool flatten (16*19*2) + attr (5)
        
        self.atchley = AtchleyEncoding()
        self.pos_enc = PositionalEncoding_1(d_model, self.joint_len)
        self.self_attn = SelfAttention(d_model)
        self.post_attn = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU())
        self.cnn = nn.Conv2d(1, 16, kernel_size=(2, 1))
        self.bn = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(2, 2)
        self.attr_embed = nn.Linear(attr_dim, d_model)
        self.classifier = nn.Sequential(nn.Linear(608 + d_model, 1), nn.Sigmoid())
        
    def recover_aa_indices(self, onehot_flat, max_len):
        onehot_reshaped = onehot_flat.view(-1, max_len, N_AA)
        aa_indices = torch.argmax(onehot_reshaped, dim=-1)
        # Set padding (all-zero one-hot) to invalid 20
        padding_mask = (onehot_reshaped.sum(dim=-1) == 0)
        aa_indices[padding_mask] = 20
        return aa_indices.long()
    
    def forward(self, x):
        seq_len_dim = (self.max_cdr3_len_input + self.max_epitope_len_input) * N_AA
        seq_input = x[:, :seq_len_dim]
        attr_input = x[:, seq_len_dim:]
        
        tcr_aa = self.recover_aa_indices(seq_input[:, :self.max_cdr3_len_input * N_AA], self.max_cdr3_len_input)
        epi_aa = self.recover_aa_indices(seq_input[:, self.max_cdr3_len_input * N_AA :], self.max_epitope_len_input)
        
        # Truncate if longer, pad with 20 if shorter
        if tcr_aa.size(1) > self.max_cdr3_len:
            tcr_aa = tcr_aa[:, :self.max_cdr3_len]
        else:
            pad_len = self.max_cdr3_len - tcr_aa.size(1)
            tcr_aa = F.pad(tcr_aa, (0, pad_len), value=20)
        
        if epi_aa.size(1) > self.max_epitope_len:
            epi_aa = epi_aa[:, :self.max_epitope_len]
        else:
            pad_len = self.max_epitope_len - epi_aa.size(1)
            epi_aa = F.pad(epi_aa, (0, pad_len), value=20)
        
        tcr_enc = self.atchley.encode(tcr_aa)  # (b, 25, 5)
        epi_enc = self.atchley.encode(epi_aa)  # (b, 15, 5)
        joint = torch.cat([epi_enc, tcr_enc], dim=1)  # (b, 40, 5)
        joint = self.pos_enc(joint)
        attn_out = self.self_attn(joint)  # (b, 40, 5)
        
        # Post-attn: apply directly on feat dim (no transpose needed)
        post = self.post_attn(attn_out)  # (b, 40, 5)
        
        # CNN: (b, 1, 40, 5)
        cnn_in = post.unsqueeze(1)
        conv_out = F.relu(self.cnn(cnn_in))  # (b, 16, 39, 5)
        bn_out = self.bn(conv_out)
        pooled = self.pool(bn_out)  # (b, 16, 19, 2)
        flattened = pooled.flatten(1)  # (b, 608)
        
        attr_feat = self.attr_embed(attr_input)  # (b, 5)
        combined = torch.cat([flattened, attr_feat], dim=1)  # (b, 613)
        return self.classifier(combined)  # (b, 1)


# --- 12. 模型十二：TITAN 模型 (Bi-LSTM + Bimodal Attention) ---
# https://github.com/PaccMann/TITAN (based on ISMB 2021 paper: https://doi.org/10.1093/bioinformatics/btab294)

class BimodalAttention(nn.Module):
    """
    TITAN Bimodal Multi-Channel Attention: multi-head cross-attention between TCR and epitope features.
    Q from TCR LSTM out, K/V from epitope LSTM out; channels for multi-head.
    """
    def __init__(self, hidden_dim, num_heads=4, dropout=0.2):
        super(BimodalAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.output_dim = hidden_dim  # For framework compatibility
        
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(2 * hidden_dim, hidden_dim)  # Cat 2*hidden -> hidden
        self.dropout = nn.Dropout(dropout)

    def forward(self, tcr_feat, epi_feat):
        # tcr_feat: (batch, tcr_len, hidden), epi_feat: (batch, epi_len, hidden)
        batch = tcr_feat.size(0)
        tcr_len = tcr_feat.size(1)
        epi_len = epi_feat.size(1)
        
        # Project
        Q = self.q_proj(tcr_feat).view(batch, tcr_len, self.num_heads, self.head_dim).transpose(1, 2)  # (b, heads, tcr_len, head_dim)
        K = self.k_proj(epi_feat).view(batch, epi_len, self.num_heads, self.head_dim).transpose(1, 2)  # (b, heads, epi_len, head_dim)
        V = self.v_proj(epi_feat).view(batch, epi_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention
        attn = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # (b, heads, tcr_len, epi_len)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        context = torch.matmul(attn, V)  # (b, heads, tcr_len, head_dim)
        context = context.transpose(1, 2).contiguous().view(batch, tcr_len, self.hidden_dim)  # (b, tcr_len, hidden)
        
        # Weighted sum over epi for each tcr position
        weights = attn.mean(dim=1)  # Average heads: (b, tcr_len, epi_len)
        epi_weighted = torch.bmm(weights, epi_feat)  # (b, tcr_len, hidden)
        
        # Cat context and weighted epi
        fused = torch.cat([context, epi_weighted], dim=-1)  # (b, tcr_len, 2*hidden)
        fused = self.out_proj(fused)  # Project back to hidden (b, tcr_len, hidden)
        
        return fused.mean(dim=1)  # Global pool over tcr_len: (b, hidden)

class TITANPredictor(nn.Module):
    """
    TITAN 整体模型：Bi-LSTM encoders for TCR/epitope + bimodal attention fusion + attr embed + MLP分类。
    基于论文：Bi-LSTM (2 layers, hidden=128, bidirectional) for seq encoding; bimodal MCA for cross-modal interaction; dense head.
    适配框架：one-hot flatten输入，recover AA indices (pad=0); attr linear cat after fusion; sigmoid output。
    """
    def __init__(self, max_cdr3_len, max_epitope_len, attr_dim, hidden_dim=128, num_layers=2, num_heads=4, dropout=0.2):
        super(TITANPredictor, self).__init__()
        
        self.max_cdr3_len = max_cdr3_len
        self.max_epitope_len = max_epitope_len
        self.attr_dim = attr_dim
        self.hidden_dim = hidden_dim * 2  # Bidirectional output
        self.unidir_hidden = hidden_dim  # Unidirectional hidden
        self.seq_feature_dim = self.hidden_dim * 2  # Fusion + attr
        
        # Embedding for AA (vocab=21: 0 pad, 1-20 AA)
        self.embedding = nn.Embedding(21, self.unidir_hidden)  # Embed to unidir_hidden=128
        
        # Bi-LSTM for TCR
        self.tcr_lstm = nn.LSTM(
            input_size=self.unidir_hidden,  # Embedding dim
            hidden_size=self.unidir_hidden,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Bi-LSTM for epitope
        self.epi_lstm = nn.LSTM(
            input_size=self.unidir_hidden,
            hidden_size=self.unidir_hidden,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Bimodal attention fusion
        self.bimodal_attn = BimodalAttention(self.hidden_dim, num_heads, dropout)
        
        # Attr embed
        self.attr_embed = nn.Linear(attr_dim, self.hidden_dim)
        
        # Classifier
        fused_dim = self.hidden_dim + self.hidden_dim
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
    def recover_aa_indices(self, onehot_flat, max_len):
        """从flatten one-hot恢复AA indices (batch, max_len)，pad=0。"""
        onehot_reshaped = onehot_flat.view(-1, max_len, 20)
        aa_indices = torch.argmax(onehot_reshaped, dim=-1) + 1  # 0-19 ->1-20
        aa_indices = torch.where(onehot_reshaped.sum(dim=-1) == 0, torch.zeros_like(aa_indices), aa_indices)
        return aa_indices.long()
    
    def forward(self, x):
        # 解包
        seq_len_dim = (self.max_cdr3_len + self.max_epitope_len) * 20
        seq_input = x[:, :seq_len_dim]
        attr_input = x[:, seq_len_dim:]
        
        # Recover indices
        tcr_aa = self.recover_aa_indices(seq_input[:, :self.max_cdr3_len * 20], self.max_cdr3_len)
        epi_aa = self.recover_aa_indices(seq_input[:, self.max_cdr3_len * 20:], self.max_epitope_len)
        
        # Embed
        tcr_emb = self.embedding(tcr_aa)  # (b, max_cdr3, unidir_hidden=128)
        epi_emb = self.embedding(epi_aa)  # (b, max_epi, 128)
        
        # TCR Bi-LSTM
        tcr_out, _ = self.tcr_lstm(tcr_emb)  # (b, max_cdr3, hidden_dim=256)
        
        # Epitope Bi-LSTM
        epi_out, _ = self.epi_lstm(epi_emb)  # (b, max_epi, 256)
        
        # Bimodal fusion
        fused_feat = self.bimodal_attn(tcr_out, epi_out)  # (b, 256)
        
        # Attr embed
        attr_feat = self.attr_embed(attr_input)  # (b, 256)
        
        # Cat + classify
        combined = torch.cat([fused_feat, attr_feat], dim=1)  # (b, 512)
        return self.classifier(combined)  # (b, 1)


# --- 13. 模型十三：PISTE 模型 (多层 Transformer 编码) ---
## https://github.com/jychen01/PISTE

class PoswiseFeedForwardNet(nn.Module):
    """
    PISTE 的位置前馈网络：Linear + ReLU + Linear + LayerNorm + residual。
    """
    def __init__(self, d_model, d_ff):
        super(PoswiseFeedForwardNet, self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=False)
        )
        self.ln = nn.LayerNorm(d_model)
        
    def forward(self, inputs):
        residual = inputs
        output = self.fc(inputs)
        return self.ln(output + residual)

class EncoderLayer(nn.Module):
    """
    PISTE 编码器层：双 Conv1d + BN + ReLU + residual。
    """
    def __init__(self, d_model):
        super(EncoderLayer, self).__init__()
        self.conv1 = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(d_model)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(d_model)

    def forward(self, enc_inputs):
        # enc_inputs: (batch, d_model, seq_len)
        x = self.conv1(enc_inputs)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = x + enc_inputs
        return self.relu(x)

class Encoder(nn.Module):
    """
    PISTE 序列编码器：one-hot Linear emb + 多层 EncoderLayer。
    """
    def __init__(self, d_model, e_layers):
        super(Encoder, self).__init__()
        self.emb = nn.Linear(N_AA, d_model)
        self.layers = nn.ModuleList([EncoderLayer(d_model) for _ in range(e_layers)])

    def forward(self, enc_inputs):
        # enc_inputs: (batch, seq_len, N_AA)
        enc_outputs = self.emb(enc_inputs)  # (batch, seq_len, d_model)
        enc_outputs = enc_outputs.transpose(1, 2)  # (batch, d_model, seq_len)
        for layer in self.layers:
            enc_outputs = layer(enc_outputs)
        enc_outputs = enc_outputs.transpose(1, 2)  # (batch, seq_len, d_model)
        return enc_outputs

class HLAEncoderLayer(nn.Module):
    """
    PISTE HLA 专用编码器层：序列位置分割 Conv + cat。
    假设 hla_seq_len=34，分割位置 0-3,4-17,18-23,24-33。
    """
    def __init__(self, d_model):
        super(HLAEncoderLayer, self).__init__()
        self.conv_block1 = EncoderLayer(d_model)
        self.conv_block2 = EncoderLayer(d_model)
        self.conv_block3 = EncoderLayer(d_model)

    def forward(self, enc_inputs):
        # enc_inputs: (batch, d_model, hla_len=34)
        hla1 = enc_inputs[:, :, 4:18]  # positions 4-17 (14)
        hla2 = torch.cat([enc_inputs[:, :, :4], enc_inputs[:, :, 18:24]], dim=-1)  # 0-3 (4) + 18-23 (6) =10
        hla3 = enc_inputs[:, :, 24:]  # 24-33 (10)
        hla1_out = self.conv_block1(hla1)
        hla2_out = self.conv_block2(hla2)
        hla3_out = self.conv_block3(hla3)
        # Reassemble: hla2[:4], hla1, hla2[4:], hla3
        enc_outputs = torch.cat([
            hla2_out[:, :, :4],
            hla1_out,
            hla2_out[:, :, 4:],
            hla3_out
        ], dim=-1)  # (batch, d_model, 4+14+6+10=34)
        return enc_outputs

class HLAEncoder(nn.Module):
    """
    PISTE HLA 编码器：one-hot emb + 多层 HLAEncoderLayer。
    """
    def __init__(self, d_model, e_layers, hla_len=34):
        super(HLAEncoder, self).__init__()
        self.hla_len = hla_len
        self.emb = nn.Linear(N_AA, d_model)
        self.layers = nn.ModuleList([HLAEncoderLayer(d_model) for _ in range(e_layers)])

    def forward(self, enc_inputs):
        # enc_inputs: (batch, hla_len, N_AA)
        enc_outputs = self.emb(enc_inputs)  # (batch, hla_len, d_model)
        enc_outputs = enc_outputs.transpose(1, 2)  # (batch, d_model, hla_len)
        for layer in self.layers:
            enc_outputs = layer(enc_outputs)
        enc_outputs = enc_outputs.transpose(1, 2)  # (batch, hla_len, d_model)
        return enc_outputs

class MultiHeadAttention(nn.Module):
    """
    PISTE 多头注意力：使用预计算 attn scores，residual + LN。
    修复：如果 attn 是 3D (batch, q_len, k_len)，扩展到 heads 维度 (batch, heads, q_len, k_len)。
    """
    def __init__(self, d_model, d, n_heads):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.d = d
        self.n_heads = n_heads
        self.w_q = nn.Linear(d_model, d * n_heads, bias=False)
        self.w_v = nn.Linear(d_model, d * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d, d_model, bias=False)
        self.ln = nn.LayerNorm(d_model)
        
    def forward(self, input_q, input_v, attn):
        # input_q: (batch, q_len, d_model), input_v: (batch, v_len, d_model), attn: (batch, q_len, k_len) or (batch, heads, q_len, k_len)
        residual, batch_size = input_q, input_q.size(0)
        q = self.w_q(input_q).view(batch_size, -1, self.n_heads, self.d).transpose(1, 2)  # (batch, heads, q_len, d)
        v = self.w_v(input_v).view(batch_size, -1, self.n_heads, self.d).transpose(1, 2)  # (batch, heads, v_len, d)
        
        # 扩展 attn 到 heads 维度如果需要
        if attn.dim() == 3:
            attn = attn.unsqueeze(1).repeat(1, self.n_heads, 1, 1)  # (batch, heads, q_len, k_len)
        
        attn = nn.Softmax(dim=-1)(attn)
        attn = torch.where(torch.isnan(attn), torch.zeros_like(attn), attn)
        context = torch.matmul(attn, v) + (1 - attn.sum(-1).unsqueeze(-1)) * q  # (batch, heads, q_len, d)
        context = context.transpose(1, 2).reshape(batch_size, -1, self.n_heads * self.d)  # (batch, q_len, heads*d)
        output = self.fc(context)
        return self.ln(output + residual), attn

class DecoderLayer(nn.Module):
    """
    PISTE 解码器层：pep_self_att (hla Q, pep V), tcr2pep_att (pep Q, tcr V), tcr2pep2hla_att (hla Q, tcr2pep V) + FFN。
    """
    def __init__(self, d_model, d, n_heads, d_ff):
        super(DecoderLayer, self).__init__()
        self.pep_self_attn = MultiHeadAttention(d_model, d, n_heads)
        self.tcr2pep_attn = MultiHeadAttention(d_model, d, n_heads)
        self.tcr2pep2hla_attn = MultiHeadAttention(d_model, d, n_heads)
        self.pep_ffn = PoswiseFeedForwardNet(d_model, d_ff)
        self.tcr2hla_ffn = PoswiseFeedForwardNet(d_model, d_ff)

    def forward(self, hla_inputs, pep_inputs, tcr_inputs, pep_attn_mask, tcr_attn_mask):
        # pep_self: Q=hla (b,hla,d), V=pep (b,pep,d), attn=(b,hla,pep)
        pep_outputs, pep_attn = self.pep_self_attn(hla_inputs, pep_inputs, pep_attn_mask)
        # tcr2pep: Q=pep (b,pep,d), V=tcr (b,tcr,d), attn=(b,pep,tcr)
        tcr2pep_outputs, _ = self.tcr2pep_attn(pep_inputs, tcr_inputs, tcr_attn_mask)
        # tcr2pep2hla: Q=hla (b,hla,d), V=tcr2pep (b,pep,d), attn=(b,hla,pep)
        tcr2pep2hla_outputs, _ = self.tcr2pep2hla_attn(hla_inputs, tcr2pep_outputs, pep_attn_mask)
        pep_outputs = self.pep_ffn(pep_outputs)
        tcr2pep2hla_outputs = self.tcr2hla_ffn(tcr2pep2hla_outputs)
        return pep_outputs, tcr2pep2hla_outputs, pep_attn

class Decoder(nn.Module):
    """
    PISTE 解码器：多层 DecoderLayer。
    """
    def __init__(self, d_model, d, n_heads, d_ff, d_layers):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList([DecoderLayer(d_model, d, n_heads, d_ff) for _ in range(d_layers)])

    def forward(self, hla_inputs, pep_inputs, tcr_inputs, pep_attn, tcr_attn):
        dec_self_attns = []
        for layer in self.layers:
            pep_outputs, tcr2hla, dec_self_attn = layer(hla_inputs, pep_inputs, tcr_inputs, pep_attn, tcr_attn)
            dec_self_attns.append(dec_self_attn)
        return pep_outputs, tcr2hla, dec_self_attns

class PISTEPredictor(nn.Module):
    """
    PISTE 模型：多层 Transformer 编码 (Conv-based) + 简化交互 (cat) + 解码器 + MLP 分类。
    基于官方 PISTE 架构，适配 one-hot 输入和属性特征。
    修复：MultiHeadAttention 中扩展 attn 到 heads 维度，确保 matmul 形状匹配。
    """
    def __init__(self, max_cdr3_len, max_epitope_len, attr_dim, d_model=64, e_layers=2, d=16, n_heads=4, d_ff=128, d_layers=2, hla_max_len=34):
        super(PISTEPredictor, self).__init__()
        
        self.max_cdr3_len = max_cdr3_len
        self.max_epitope_len = max_epitope_len
        self.attr_dim = attr_dim
        self.d_model = d_model
        self.pep_len = max_epitope_len  # 使用实际 max_epitope_len
        self.tcr_len = max_cdr3_len     # 使用实际 max_cdr3_len
        self.hla_len = hla_max_len     # 固定 HLA 长度
        
        # 编码器
        self.pep_encoder = Encoder(d_model, e_layers)
        self.hla_encoder = HLAEncoder(d_model, e_layers, self.hla_len)
        self.tcr_encoder = Encoder(d_model, e_layers)
        
        # 解码器
        self.decoder = Decoder(d_model, d, n_heads, d_ff, d_layers)
        
        # HLA from attr proj to padded seq (attr_dim -> N_AA * hla_len)
        self.hla_proj = nn.Linear(attr_dim, N_AA * self.hla_len)
        
        # 分类头 (3 * hla_len * d_model)
        input_dim = 3 * self.hla_len * d_model
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        self.seq_feature_dim = input_dim
        
    def forward(self, x):
        batch_size = x.shape[0]
        seq_len_dim = (self.tcr_len + self.pep_len) * N_AA  # 现在匹配实际 max_cdr3 + max_epitope
        seq_input = x[:, :seq_len_dim]
        attr_input = x[:, seq_len_dim:]  # 维度 = attr_dim
        
        # Epitope (pep)
        pep_seq_flat = seq_input[:, :self.pep_len * N_AA]
        pep_seq = pep_seq_flat.view(batch_size, self.pep_len, N_AA)
        
        # TCR (CDR3)
        tcr_seq_flat = seq_input[:, self.pep_len * N_AA : ]
        tcr_seq = tcr_seq_flat.view(batch_size, self.tcr_len, N_AA)
        
        # MHC (hla) from attr
        hla_flat = self.hla_proj(attr_input)  # (batch, N_AA * hla_len)
        hla_seq = hla_flat.view(batch_size, self.hla_len, N_AA)
        
        # 编码
        pep_enc = self.pep_encoder(pep_seq)  # (batch, pep_len, d_model)
        hla_enc = self.hla_encoder(hla_seq)  # (batch, hla_len, d_model)
        tcr_enc = self.tcr_encoder(tcr_seq)  # (batch, tcr_len, d_model)
        
        # 简化交互：uniform attn masks
        pep_attn = torch.ones(batch_size, self.hla_len, self.pep_len, device=x.device) / self.pep_len
        tcr_attn = torch.ones(batch_size, self.pep_len, self.tcr_len, device=x.device) / self.tcr_len
        
        # 解码
        pep_out, tcr2hla_out, _ = self.decoder(hla_enc, pep_enc, tcr_enc, pep_attn, tcr_attn)
        # pep_out (batch, hla_len, d_model), tcr2hla_out (batch, hla_len, d_model)
        
        # 融合 (cat along seq dim)
        combined = torch.cat([tcr2hla_out, pep_out, hla_enc], dim=1)  # (batch, 3*hla_len, d_model)
        combined_flat = combined.view(batch_size, -1)
        
        return self.classifier(combined_flat)

# --- 14. 模型十四：TPepRet 模型 (Transformer 编码 + Feature Fusion) ---
## https://github.com/CSUBioGroup/TPepRet


class PositionalEncoding(nn.Module):
    """
    位置编码模块 (sinusoidal)，用于 Transformer 输入。
    """
    def __init__(self, d_model: int, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (seq_len, batch, d_model)
        return x + self.pe[:x.size(0), :]

class TPepRetEncoder(nn.Module):
    """
    TPepRet 风格 Transformer 编码器：Embedding + Positional + Multi-layer Transformer。
    """
    def __init__(self, max_len: int, d_model: int = 128, nhead: int = 8, num_layers: int = 4, 
                 dim_feedforward: int = 512, dropout: float = 0.1):
        super(TPepRetEncoder, self).__init__()
        self.d_model = d_model
        self.embedding = nn.Linear(N_AA, d_model)
        self.pos_encoder = PositionalEncoding(d_model=d_model, max_len=max_len)
        
        # 使用关键字参数避免参数冲突
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout, 
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer, 
            num_layers=num_layers
        )
        
        self.output_dim = d_model  # 每个序列的 pooled 输出维度

    def forward(self, seq_input):
        # seq_input: (batch, seq_len, N_AA)
        embedded = self.embedding(seq_input) * math.sqrt(self.d_model)  # Scale embedding
        # Add pos: transpose to seq_first for pos_encoder, then back
        pos_added = self.pos_encoder(embedded.transpose(0, 1)).transpose(0, 1)  # (batch, seq_len, d_model)
        encoded = self.transformer(pos_added)
        # Global average pool over seq_len
        pooled = encoded.mean(dim=1)  # (batch, d_model)
        return pooled

class TPepRetPredictor(nn.Module):
    """
    TPepRet 整体模型：独立 Transformer 编码 TCR 和 Epitope + Cross-Attention 融合 + 属性嵌入 + MLP。
    基于原始 TPepRet 架构，适配 one-hot 输入。
    """
    def __init__(self, max_cdr3_len: int, max_epitope_len: int, attr_dim: int, 
                 d_model: int = 128, nhead: int = 8, num_layers: int = 4):
        super(TPepRetPredictor, self).__init__()
        
        self.max_cdr3_len = max_cdr3_len
        self.max_epitope_len = max_epitope_len
        self.attr_dim = attr_dim
        self.d_model = d_model
        
        # 独立编码器（使用关键字参数调用）
        self.tcr_encoder = TPepRetEncoder(
            max_len=max_cdr3_len, 
            d_model=d_model, 
            nhead=nhead, 
            num_layers=num_layers
        )
        self.epi_encoder = TPepRetEncoder(
            max_len=max_epitope_len, 
            d_model=d_model, 
            nhead=nhead, 
            num_layers=num_layers
        )
        
        # 跨注意力融合：TCR as query, Epitope as key/value
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, 
            num_heads=nhead, 
            batch_first=True
        )
        
        # 属性嵌入
        self.attr_embed = nn.Linear(attr_dim, d_model)
        
        # 融合维度：cross_out (d_model) + attr (d_model) + avg(tcr, epi) (d_model) = 3 * d_model
        self.seq_feature_dim = d_model * 3
        
        # Classifier MLP (原始：3层，with GELU)
        self.classifier = nn.Sequential(
            nn.Linear(self.seq_feature_dim, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # 解包输入（使用 N_AA 常量）
        seq_len_dim = (self.max_cdr3_len + self.max_epitope_len) * N_AA
        seq_input = x[:, :seq_len_dim]
        attr_input = x[:, seq_len_dim:]
        
        # 重塑序列
        cdr3_seq = seq_input[:, :self.max_cdr3_len * N_AA].view(-1, self.max_cdr3_len, N_AA)
        epi_seq = seq_input[:, self.max_cdr3_len * N_AA : seq_len_dim].view(-1, self.max_epitope_len, N_AA)
        
        # 编码
        tcr_embed = self.tcr_encoder(cdr3_seq)  # (batch, d_model)
        epi_embed = self.epi_encoder(epi_seq)  # (batch, d_model)
        
        # 跨注意力：TCR query, Epi key/value (pooled as single token)
        tcr_query = tcr_embed.unsqueeze(1)  # (batch, 1, d_model)
        epi_key_value = epi_embed.unsqueeze(1)
        cross_out, _ = self.cross_attn(
            query=tcr_query, 
            key=epi_key_value, 
            value=epi_key_value
        )
        cross_out = cross_out.squeeze(1)  # (batch, d_model)
        
        # 属性嵌入
        attr_embed = self.attr_embed(attr_input)
        
        # 融合：cross + attr + mean(tcr, epi)
        fused = torch.cat([cross_out, attr_embed, (tcr_embed + epi_embed) / 2], dim=1)  # (batch, 3 * d_model)
        
        return self.classifier(fused)



###添加一个简单的检索层（如余弦相似度匹配预训练嵌入），但这可能增加复杂性并影响公平性（其他模型无检索）
# class TPepRetPredictor(nn.Module):
#     """
#     TPepRet 模型：多层 Transformer 编码，用于 TCR-pMHC 结合预测。
#     - 核心：Transformer 编码器处理连接的 Epitope + CDR3_beta 序列。
#     - 属性（V_beta, J_beta, MHC）使用嵌入层。
#     - 添加简单检索机制：使用余弦相似度匹配预计算的肽嵌入数据库。
#     - 融合后通过 MLP 分类。
#     - 注意：检索数据库（pep_embed_db）需预加载实际嵌入；在示例中随机初始化。
#     """
#     def __init__(self, max_cdr3_len, max_epitope_len, attr_dim, token_emb=128, num_heads=8, num_layers=3, dim_feedforward=256, dropout=0.1, db_size=1000):
#         super(TPepRetPredictor, self).__init__()
#         self.max_cdr3_len = max_cdr3_len
#         self.max_epitope_len = max_epitope_len
#         self.attr_dim = attr_dim
        
#         # 令牌嵌入（21: 20氨基酸 + pad=0）
#         self.token_embedding = nn.Embedding(21, token_emb, padding_idx=0)
#         self.pos_embed = nn.Parameter(torch.zeros(1, max_cdr3_len + max_epitope_len + 1, token_emb))  # +1 for CLS
#         self.cls_token = nn.Parameter(torch.zeros(1, 1, token_emb))
#         nn.init.trunc_normal_(self.pos_embed, std=0.02)
#         nn.init.trunc_normal_(self.token_embedding.weight, std=0.02)
        
#         # Transformer 编码器（使用位置参数顺序）
#         encoder_layer = nn.TransformerEncoderLayer(token_emb, num_heads, dim_feedforward, dropout, "gelu")
#         self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
#         self.norm = nn.LayerNorm(token_emb)
        
#         # 属性嵌入
#         self.gene_embed = nn.Linear(attr_dim, token_emb)
        
#         # 检索数据库（预计算肽嵌入,ref:https://github.com/CSUBioGroup/TPepRet/blob/main/TPepRet_train/codes/embedding_protein.txt）
#         embed_lines = []
        

#         with open('/home/dengyang/code/My_code/embedding_protein.txt', 'r') as f:
#           for line_num, line in enumerate(f, 1):
#             # 移除所有 '[' 和 ']'，并 strip 额外空格
#             cleaned_line = line.strip().replace('[', '').replace(']', '').strip()
#             # 按空格拆分，并过滤空字符串
#             embeds_str = [x for x in cleaned_line.split() if x]
#             # 转换为 float
#             embeds = [float(x) for x in embeds_str]
#             embed_lines.append(embeds)         
        
#         self.pep_embed_db = nn.Parameter(torch.tensor(embed_lines, dtype=torch.float32))  # (num_embeds, embed_dim)
        
#         # 添加投影层：将文件维度投影到 token_emb
#         file_dim = len(embed_lines[0])
#         self.db_proj = nn.Linear(file_dim, token_emb)
        
#         # 序列特征维度
#         self.seq_feature_dim = token_emb  # Transformer 输出
#         total_input_dim = token_emb + token_emb + 1  # pooled + attr + retrieval sim (标量1维)
        
#         # 分类器 MLP
#         self.classifier = nn.Sequential(
#             nn.Linear(total_input_dim, 256),
#             nn.ReLU(),
#             nn.Dropout(0.3),
#             nn.Linear(256, 128),
#             nn.ReLU(),
#             nn.Dropout(0.3),
#             nn.Linear(128, 1),
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         seq_len_dim = (self.max_cdr3_len + self.max_epitope_len) * 20  # N_AA=20
#         seq_flat = x[:, :seq_len_dim]
#         attr_input = x[:, seq_len_dim:]
        
#         # one-hot 平展到 token 索引
#         cdr3_tokens = onehot_flat_to_indices(seq_flat[:, :self.max_cdr3_len * 20], self.max_cdr3_len)
#         epi_tokens = onehot_flat_to_indices(seq_flat[:, self.max_cdr3_len * 20:], self.max_epitope_len)
        
#         # 连接序列
#         seq = torch.cat([epi_tokens, cdr3_tokens], dim=1)  # (batch, L_seq)
#         bsz, L = seq.size()
        
#         # 嵌入 + 位置 + CLS
#         emb = self.token_embedding(seq) * math.sqrt(self.token_embedding.embedding_dim)
#         cls_tokens = self.cls_token.expand(bsz, -1, -1)
#         emb = torch.cat([cls_tokens, emb], dim=1) + self.pos_embed[:, :L + 1, :]
        
#         # Transformer
#         h = self.transformer(emb)
#         h = self.norm(h)
#         pooled = h[:, 0, :]  # CLS token as pooled
        
#         # 属性嵌入
#         gene_embed = self.gene_embed(attr_input)
        
#         # 检索机制：提取 epi 嵌入（平均 epi 部分）
#         epi_emb = h[:, 1:1 + self.max_epitope_len, :].mean(dim=1)  # (batch, token_emb)
        
#         # 投影数据库嵌入到 token_emb 维度（即使匹配，也通过 Identity）
#         pep_db_proj = self.db_proj(self.pep_embed_db)  # (db_size, token_emb)
        
        
#         # 余弦相似度与数据库匹配
#         sim = F.cosine_similarity(epi_emb.unsqueeze(1), pep_db_proj.unsqueeze(0), dim=-1)  # (batch, db_size)
#         top_sim = sim.topk(1, dim=1).values.squeeze(-1)  # (batch,) top similarity score
        
#         # 融合（pooled + attr + top_sim）
#         combined = torch.cat([pooled, gene_embed, top_sim.unsqueeze(1)], dim=1)
        
#         return self.classifier(combined)    




# --- 15. 模型十五：TCRBagger 模型 (CNN 特征提取 + MIL Attention 聚合) ---
# https://github.com/bm2-lab/TCRBagger (TensorFlow -> PyTorch 转换)

class MILAttention(nn.Module):
    """
    简单 MIL 注意机制：对 bag 实例特征计算 attention weights 并聚合。
    忠实原始：tanh activation + softmax weights, weighted sum。
    """
    def __init__(self, feature_dim: int, hidden_dim: int = 32):
        super(MILAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, features):
        # features: (batch, bag_size, feature_dim)；这里 bag_size=1 for single instance
        att_weights = self.attention(features).squeeze(-1)  # (batch, bag_size)
        att_weights = F.softmax(att_weights, dim=-1)
        aggregated = torch.sum(att_weights.unsqueeze(-1) * features, dim=1)  # (batch, feature_dim)
        return aggregated

class TCRBaggerPredictor(nn.Module):
    """
    TCRBagger 基础模型：Bi-LSTM 编码器，用于 TCR-pMHC 结合预测。
    - 核心：双向 LSTM 对 CDR3_beta 和 Epitope 序列进行编码。
    - 属性（V_beta, J_beta, MHC）使用嵌入层。
    - 融合后通过 MLP 分类。
    - 注意：bagging 集成（多个模型平均）在训练/测试时外部处理（见 main.py）。
    """
    def __init__(self, max_cdr3_len, max_epitope_len, attr_dim, hidden_dim=128, embed_dim=64):
        super(TCRBaggerPredictor, self).__init__()
        self.max_cdr3_len = max_cdr3_len
        self.max_epitope_len = max_epitope_len
        self.attr_dim = attr_dim
        
        # Bi-LSTM 编码器（双向）
        self.cdr3_lstm = nn.LSTM(input_size=20, hidden_size=hidden_dim, bidirectional=True, batch_first=True)  # N_AA=20
        self.epi_lstm = nn.LSTM(input_size=20, hidden_size=hidden_dim, bidirectional=True, batch_first=True)
        
        # 属性嵌入（V/J beta + MHC）
        self.gene_embed = nn.Linear(attr_dim, embed_dim)
        
        # 序列特征维度：hidden_dim * 2 (bi-dir) * 2 (两个序列)
        self.seq_feature_dim = hidden_dim * 2 * 2
        total_input_dim = self.seq_feature_dim + embed_dim
        
        # 分类器 MLP
        self.classifier = nn.Sequential(
            nn.Linear(total_input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        seq_len_dim = (self.max_cdr3_len + self.max_epitope_len) * 20  # N_AA=20
        seq_input = x[:, :seq_len_dim]
        attr_input = x[:, seq_len_dim:]
        
        # 重塑序列
        cdr3_input = seq_input[:, :self.max_cdr3_len * 20].view(-1, self.max_cdr3_len, 20)
        epi_input = seq_input[:, self.max_cdr3_len * 20:].view(-1, self.max_epitope_len, 20)
        
        # Bi-LSTM 编码
        cdr3_out, _ = self.cdr3_lstm(cdr3_input)
        epi_out, _ = self.epi_lstm(epi_input)
        
        # 取平均或最后隐藏状态
        cdr3_feat = cdr3_out.mean(dim=1)  # (batch, hidden*2)
        epi_feat = epi_out.mean(dim=1)    # (batch, hidden*2)
        
        # 属性嵌入
        gene_embed = self.gene_embed(attr_input)
        
        # 融合
        combined = torch.cat([cdr3_feat, epi_feat, gene_embed], dim=1)
        
        return self.classifier(combined)
        
        
        
# --- 16. 模型十六：TEINet 模型 (预训练 RNN 编码 + Concat + MLP 投影) ---
# https://github.com/jiangdada1221/TEINet (模拟 TCRpeg RNN 编码器)

class TEINetEncoder(nn.Module):
    """
    模拟 TCRpeg 风格 RNN 编码器：Embedding + LSTM + last hidden (64D)。
    忠实原始：处理变长序列，但这里固定 max_len with padding implicit in one-hot。
    """
    def __init__(self, max_len: int, embed_dim: int = 32, hidden_dim: int = 64, dropout: float = 0.1):
        super(TEINetEncoder, self).__init__()
        self.embedding = nn.Linear(N_AA, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=1, batch_first=True, dropout=dropout)
        self.output_dim = hidden_dim

    def forward(self, seq_input):
        # seq_input: (batch, seq_len, N_AA)
        embedded = self.embedding(seq_input)  # (batch, seq_len, embed_dim)
        _, (hn, _) = self.lstm(embedded)  # hn: (1, batch, hidden_dim)
        return hn.squeeze(0)  # (batch, hidden_dim=64)

class TEINetPredictor(nn.Module):
    """
    TEINet 整体模型：独立 RNN 编码 TCR 和 Epitope + LayerNorm + Concat + MLP 投影。
    基于原始 TEINet 架构，适配 one-hot 输入，无需字符串 seqs。
    """
    def __init__(self, max_cdr3_len: int, max_epitope_len: int, attr_dim: int, 
                 cat_size: int = 64, dropout: float = 0.1, normalize: bool = True, weight_decay: float = 0.0):
        super(TEINetPredictor, self).__init__()
        
        self.max_cdr3_len = max_cdr3_len
        self.max_epitope_len = max_epitope_len
        self.attr_dim = attr_dim
        self.cat_size = cat_size  # emb_dim per encoder
        self.normalize = normalize
        self.weight_decay = weight_decay
        self.device = DEVICE
        
        # 独立编码器 (模拟 en_tcr / en_epi.model)
        self.tcr_encoder = TEINetEncoder(max_cdr3_len, dropout=dropout)
        self.epi_encoder = TEINetEncoder(max_epitope_len, dropout=dropout)
        
        # LayerNorm (修正为 full emb_dim=64)
        self.layer_norm_tcr = nn.LayerNorm(cat_size)
        self.layer_norm_epi = nn.LayerNorm(cat_size)
        self.attr_embed = nn.Linear(attr_dim, cat_size)  # 添加属性嵌入
        
        # 投影头 (cat_size*3 = 192 输入；忠实原始 Sequential，但加 attr)
        concat_dim = cat_size * 3  # tcr + epi + attr
        self.projection = nn.Sequential(
            nn.Linear(concat_dim, cat_size),  # 192 -> 64
            nn.Dropout(dropout),
            nn.SELU(),
            nn.Linear(cat_size, cat_size // 4),  # 64 -> 16
            nn.Dropout(dropout),
            nn.SELU(),
            nn.Linear(cat_size // 4, 1),  # 16 -> 1
            nn.Sigmoid()  # 添加以匹配二分类
        )
        
        self.seq_feature_dim = concat_dim
        
    def forward(self, x):
        # 解包输入
        seq_len_dim = (self.max_cdr3_len + self.max_epitope_len) * N_AA
        seq_input = x[:, :seq_len_dim]
        attr_input = x[:, seq_len_dim:]
        
        # 重塑序列
        cdr3_seq = seq_input[:, :self.max_cdr3_len * N_AA].view(-1, self.max_cdr3_len, N_AA)
        epi_seq = seq_input[:, self.max_cdr3_len * N_AA : seq_len_dim].view(-1, self.max_epitope_len, N_AA)
        
        # 编码 (模拟 _get_emb)
        tcr_emb = self.tcr_encoder(cdr3_seq)  # (batch, 64)
        epi_emb = self.epi_encoder(epi_seq)  # (batch, 64)
        
        # 正则化 (L2 on norms)
        regularization = self.weight_decay * (tcr_emb.norm(dim=1).pow(2).sum() + epi_emb.norm(dim=1).pow(2).sum())
        
        # Normalize if enabled
        if self.normalize:
            tcr_emb = self.layer_norm_tcr(tcr_emb)
            epi_emb = self.layer_norm_epi(epi_emb)
        
        # 属性嵌入
        attr_emb = self.attr_embed(attr_input)
        
        # Concat (tcr + epi + attr)
        cat = torch.cat((tcr_emb, epi_emb, attr_emb), dim=-1)  # (batch, 192)
        
        pred = self.projection(cat)
        
        if self.weight_decay == 0.0:
            return pred
        else:
            return pred, regularization


# --- 18. 模型十七：DLpTCR 模型 (Ensemble: FULL MLP + CNN + ResNet1D) ---
# https://github.com/jiangBiolab/DLpTCR (Keras ensemble -> PyTorch)

class FullMLP(nn.Module):
    """
    FULL base model: MLP on PCA-projected features.
    """
    def __init__(self, input_dim, pca_dim=18, hidden_dim=128, dropout=0.2):
        super(FullMLP, self).__init__()
        self.pca_proj = nn.Linear(input_dim, pca_dim)
        self.mlp = nn.Sequential(
            nn.Linear(pca_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x, seq_len_dim):
        pca_feat = self.pca_proj(x)
        return self.mlp(pca_feat)

class CNNBase(nn.Module):
    """
    CNN base model: Conv1D stack on projected one-hot seq (fixed len=20).
    """
    def __init__(self, seq_len_dim, attr_dim, max_cdr3_len, max_epitope_len, pca_dim=20, dropout=0.25):
        super(CNNBase, self).__init__()
        self.max_len = 20
        self.seq_proj = nn.Linear(seq_len_dim, self.max_len * N_AA)
        self.conv1 = nn.Conv1d(N_AA, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.flatten = nn.Flatten()
        # Exact feat dim after 3 pools: len=20 ->10->5->2, 128*2=256
        cnn_feat_dim = 128 * (self.max_len // 8)
        self.dense = nn.Sequential(
            nn.Linear(cnn_feat_dim + pca_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        self.pca_embed = nn.Linear(attr_dim, pca_dim)

    def forward(self, x, seq_len_dim):
        seq_input = x[:, :seq_len_dim]
        fixed_seq_flat = self.seq_proj(seq_input)
        fixed_seq = fixed_seq_flat.view(-1, N_AA, self.max_len)
        feat1 = F.relu(self.pool(self.conv1(fixed_seq)))
        feat2 = F.relu(self.pool(self.conv2(feat1)))
        feat3 = F.relu(self.pool(self.conv3(feat2)))
        flat = self.flatten(feat3)
        attr_pca = self.pca_embed(x[:, seq_len_dim:])
        combined = torch.cat([flat, attr_pca], dim=1)
        return self.dense(combined)

class ResNetBlock(nn.Module):
    """
    Residual block for 1D ResNet.
    """
    def __init__(self, channels, kernel_size=3):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=1)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=1)
        self.bn2 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return self.relu(out)

class ResNetBase(nn.Module):
    """
    RESNET base model: Residual blocks on PCA seq-like features.
    """
    def __init__(self, attr_dim, pca_dim=10, channels=64, num_blocks=3, dropout=0.2):
        super(ResNetBase, self).__init__()
        self.pca_proj = nn.Linear(attr_dim, pca_dim * 20)
        self.conv_init = nn.Conv1d(pca_dim, channels, kernel_size=3, padding=1)
        self.blocks = nn.ModuleList([ResNetBlock(channels) for _ in range(num_blocks)])
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.dense = nn.Sequential(
            nn.Linear(channels, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x, seq_len_dim):
        attr = x[:, seq_len_dim:]
        pca_seq = self.pca_proj(attr).view(-1, 10, 20)  # Fixed reshape
        feat = F.relu(self.conv_init(pca_seq))
        for block in self.blocks:
            feat = block(feat)
        pooled = self.global_pool(feat).squeeze(-1)
        return self.dense(pooled)

class DLpTCRPredictor(nn.Module):
    """
    DLpTCR 整体模型：Ensemble of FULL, CNN, RESNET for β-chain TCR-peptide prediction.
    Average probs for final binding score. Based on paper/repo, adapted to one-hot input.
    """
    def __init__(self, max_cdr3_len, max_epitope_len, attr_dim):
        super(DLpTCRPredictor, self).__init__()
        
        self.max_cdr3_len = max_cdr3_len
        self.max_epitope_len = max_epitope_len
        self.attr_dim = attr_dim
        self.seq_len_dim = (max_cdr3_len + max_epitope_len) * N_AA
        self.input_dim = self.seq_len_dim + attr_dim
        
        # Base models
        self.full = FullMLP(self.input_dim, pca_dim=18)
        self.cnn = CNNBase(self.seq_len_dim, self.attr_dim, max_cdr3_len, max_epitope_len, pca_dim=20)
        self.resnet = ResNetBase(self.attr_dim, pca_dim=10)
        
        self.seq_feature_dim = 1  # Ensemble output
        
    def forward(self, x):
        # Unpack
        seq_input = x[:, :self.seq_len_dim]
        attr_input = x[:, self.seq_len_dim:]
        
        # Base predictions
        p_full = self.full(torch.cat([seq_input, attr_input], dim=1), self.seq_len_dim)
        p_cnn = self.cnn(x, self.seq_len_dim)
        p_resnet = self.resnet(x, self.seq_len_dim)
        
        # Ensemble average
        ensemble_p = (p_full + p_cnn + p_resnet) / 3
        return ensemble_p
                


# --- 18. 模型十八：PRIME 2.0 模型 (5-layer MLP for TCR Recognition) ---  # 修改: 简化到纯MLP
# https://github.com/GfellerLab/PRIME (MLP from PRIME2 dir/paper; R mlp -> PyTorch)
class PRIME2Predictor(nn.Module):
    def __init__(self, max_cdr3_len, max_epitope_len, attr_dim,
                 token_emb=128, nhead=8, num_layers=2, dim_feedforward=256, dropout=0.1, feat_mode='atchley'):
        super().__init__()
        self.max_cdr3_len = max_cdr3_len
        self.max_epitope_len = max_epitope_len
        self.attr_dim = attr_dim
        self.feat_mode = feat_mode

        # Token embedding and projection
        self.token_embedding = nn.Embedding(21, token_emb, padding_idx=0)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, token_emb))
        self.pos_embed = nn.Parameter(torch.zeros(1, max_cdr3_len + max_epitope_len + 1, token_emb))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.token_embedding.weight, std=0.02)

        # Transformer Encoder Layer
        encoder_layer = nn.TransformerEncoderLayer(d_model=token_emb, nhead=nhead,
                                                   dim_feedforward=dim_feedforward, dropout=dropout, activation="gelu", 
                                                   batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(token_emb)

        # Modified: Remove inter_cnn and attn_pool; direct to 5-layer MLP as per original focus
        self.output_net = nn.Sequential(  # 5-layer MLP
            nn.Linear(token_emb + attr_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        seq_len_dim = (self.max_cdr3_len + self.max_epitope_len) * N_AA
        seq_flat = x[:, :seq_len_dim]
        attr_input = x[:, seq_len_dim:]

        cdr3_tokens = onehot_flat_to_indices(seq_flat[:, :self.max_cdr3_len * N_AA], self.max_cdr3_len)
        epi_tokens = onehot_flat_to_indices(seq_flat[:, self.max_cdr3_len * N_AA:], self.max_epitope_len)

        seq = torch.cat([epi_tokens, cdr3_tokens], dim=1)
        bsz, L = seq.size()

        emb = self.token_embedding(seq) * math.sqrt(self.token_embedding.embedding_dim)
        cls_tokens = self.cls_token.expand(bsz, -1, -1)
        emb = torch.cat([cls_tokens, emb], dim=1) + self.pos_embed[:, :L + 1, :]

        h = self.transformer(emb)
        h = self.norm(h)
        pooled = h.mean(dim=1)  # Modified: Simple mean pool instead of attn_pool

        combined = torch.cat([pooled, attr_input], dim=1)
        return self.output_net(combined)


# --- 19. 通用训练和测试函数 ---

def train_model(model: nn.Module, X_train: np.ndarray, Y_train: np.ndarray, params: dict):
    """通用的 GPU 训练函数。"""
    
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)
    Y_train_tensor = torch.tensor(Y_train, dtype=torch.float32).view(-1, 1).to(DEVICE)
    
    model.to(DEVICE)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), 
                                 lr=params['learning_rate'], 
                                 weight_decay=params['weight_decay'])
    
    train_dataset = torch.utils.data.TensorDataset(X_train_tensor, Y_train_tensor)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)

    print(f"开始训练，Epochs={params['epochs']}, BatchSize={params['batch_size']}, Device={DEVICE}")
    for epoch in range(params['epochs']):
        model.train()
        total_loss = 0
        for X_batch, Y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, Y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * X_batch.size(0)
        
        avg_loss = total_loss / len(X_train)
        if (epoch + 1) % 5 == 0 or epoch == 0:
             print(f'Epoch [{epoch+1}/{params["epochs"]}], Avg Loss: {avg_loss:.4f}')
    return model


def test_model(models: nn.Module or list, X_test: np.ndarray, Y_test: np.ndarray, model_name: str, dataset_name: str):
    """
    在测试集上进行预测，计算/保存性能指标和曲线数据。支持list for ensemble.
    """

    if not isinstance(models, list):
        models = [models]

    dataset_folder = f"{dataset_name}/"
    os.makedirs(dataset_folder, exist_ok=True)
    
    # 分批处理 X_test 以避免 OOM
    batch_size = 64  # 可调整，根据 GPU 内存
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
    test_dataset = torch.utils.data.TensorDataset(X_test_tensor)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    with torch.no_grad():
        torch.cuda.empty_cache()
        probs = []
        for batch in test_loader:
            X_batch = batch[0]  # DataLoader 返回 tuple
            batch_probs = []
            for model in models:
                model.to(DEVICE)
                model.eval()
                prob = model(X_batch).squeeze().cpu().numpy()
                batch_probs.append(prob)
            avg_prob = np.mean(batch_probs, axis=0)  # 如果 ensemble，平均；否则单一
            probs.append(avg_prob)
        Y_pred_proba = np.concatenate(probs)  # 合并所有 batch 的预测
    
    Y_pred_class = (Y_pred_proba >= 0.5).astype(int)
    Y_test = Y_test.astype(int)
    
    

    # 计算 scalar 指标 (单次)
    auc_roc = roc_auc_score(Y_test, Y_pred_proba)
    accuracy = accuracy_score(Y_test, Y_pred_class)
    precision = precision_score(Y_test, Y_pred_class, zero_division=0)
    recall = recall_score(Y_test, Y_pred_class, zero_division=0)
    f1 = f1_score(Y_test, Y_pred_class, zero_division=0)
    precisions, recalls, _ = precision_recall_curve(Y_test, Y_pred_proba)  # _ = thresholds_pr
    auprc = auc(recalls, precisions)
    
    # --- 1. 性能指标评估 (单次) ---
    print("\n" + "~"*50)
    print(f"【{model_name} 性能指标 (数据集: {dataset_name})】")
    print("~"*50)
    print(f"测试集样本总数: {len(X_test)}")
    print(f"AUC (ROC): {auc_roc:.4f}")
    print(f"PRAUC: {auprc:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    # --- 2. 预测结果保存到 CSV ---
    results_df = pd.DataFrame({
        'Y_True': Y_test,
        'Y_Predicted_Proba': Y_pred_proba,
        'Y_Predicted_Class': Y_pred_class
    })
    predictions_filename = f"{dataset_folder}{model_name}_predictions.csv"
    results_df.to_csv(predictions_filename, index=False)
    
    # --- 3. 性能指标保存到 CSV (scalar only) ---
    metrics_summary_df = pd.DataFrame({
        'Model': [model_name],
        'Test_Samples': [len(X_test)],
        'AUC_ROC': [auc_roc],
        'PRAUC': [auprc],
        'Accuracy': [accuracy],
        'Precision': [precision],
        'Recall': [recall],
        'F1_Score': [f1]
    })
    metrics_filename = f"{dataset_folder}{model_name}_metrics.csv"
    metrics_summary_df.to_csv(metrics_filename, index=False)
    
    # --- 4. 新增：保存 ROC 曲线数据 (只 FPR/TPR, 忽略 thresholds) ---
    fpr, tpr, _ = roc_curve(Y_test, Y_pred_proba)
    roc_df = pd.DataFrame({
        'FPR': fpr,
        'TPR': tpr
    })
    roc_filename = f"{dataset_folder}{model_name}_roc_data.csv"
    roc_df.to_csv(roc_filename, index=False)
    
    # --- 5. 新增：保存 PR 曲线数据 (只 Precision/Recall, 忽略 thresholds) ---
    pr_df = pd.DataFrame({
        'Precision': precisions,
        'Recall': recalls
    })
    pr_filename = f"{dataset_folder}{model_name}_pr_data.csv"
    pr_df.to_csv(pr_filename, index=False)
    
    print("-" * 50)
    print(f"✅ 结果已保存到文件夹: {dataset_folder}")
    print(f"✅ 预测结果: {predictions_filename}")
    print(f"✅ 性能指标: {metrics_filename}")
    print(f"✅ ROC 数据: {roc_filename}")
    print(f"✅ PR 数据: {pr_filename}")