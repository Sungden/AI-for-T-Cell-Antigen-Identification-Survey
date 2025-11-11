# AI for T-Cell Antigen Identification: Data Resources, Computational Methods, and  Benchmarking
AI-Driven Computational Methods, Data Resources, and Benchmarking for T-Cell Antigen Identification

![awesome](https://img.shields.io/badge/awesome-Survey-green) ![License](https://img.shields.io/badge/License-MIT-blue)

## Available Datasets for T-cell Antigen Identification

### Peptide-MHC I Binding

| Dataset name | Published in | Resources |
|-------|--------------|-----------|
| [SYFPEITHI](https://link.springer.com/article/10.1007/S002510050595) | Immunogenetics, 1999 | [Code](http://www.syfpeithi.de/) |
| [MHCBN](https://academic.oup.com/bioinformatics/article/19/5/665/239775) | Bioinformatics, 2003 | [Code](http://www.imtech.res.in/raghava/mhcbn) |
| [EPIMHC](https://academic.oup.com/bioinformatics/article/21/9/2140/409020) | Bioinformatics, 2005 | [Code](http://immunax.dfci.harvard.edu/bioinformatics/epimhc/) |
| [Abelin et al.](https://www.cell.com/immunity/fulltext/S1074-7613(17)30042-0) | Immunity, 2017 | [Code](https://www.cell.com/immunity/fulltext/S1074-7613(17)30042-0#mmc1) |
| [IEDB](https://academic.oup.com/nar/article/47/D1/D339/5144151) | Nucl. Acids Res., 2019 | [Code](https://www.iedb.org/) |
| [Sarkizova et al.](https://www.nature.com/articles/s41587-019-0322-9) | Nat. Biotechnol., 2020 | [Code](https://massive.ucsd.edu/ProteoSAFe/static/massive.jsp?redirect=auth) |

### Peptide-MHC II Binding

| Dataset name | Published in | Resources |
|-------|--------------|-----------|
| [VDJdb](https://academic.oup.com/nar/article/46/D1/D419/4101254) | Nucl. Acids Res., 2018 | [Code](https://vdjdb.cdr3.net/) |
| [IEDB](https://academic.oup.com/nar/article/47/D1/D339/5144151) | Nucl. Acids Res., 2019 | [Code](https://www.iedb.org/) |
| [Rappazzo et al.](https://www.nature.com/articles/s41467-020-18204-2) | Nat. Commun., 2020 | [Code](https://www.nature.com/articles/s41467-020-18204-2#Sec24) |
| [Strazar et al.](https://www.cell.com/immunity/fulltext/S1074-7613(23)00226-1?uuid=uuid%3Af753c662-5258-4ab0-8c3c-b0f62d2d2cfb) | Immunity, 2023 | [Code](https://www.cell.com/immunity/fulltext/S1074-7613(23)00226-1) |

### TCR-pMHC Binding

| Dataset name | Published in | Resources |
|-------|--------------|-----------|
| [BindingDB](https://academic.oup.com/nar/article/44/D1/D1045/2502601) | Nucl. Acids Res., 2016 | [Code](https://www.bindingdb.org/) |
| [McPAS-TCR](https://academic.oup.com/bioinformatics/article/33/18/2924/3803440) | Bioinformatics, 2017 | [Code](http://friedmanlab.weizmann.ac.il/McPAS-TCR/) |
| [Dash et al.](https://www.nature.com/articles/nature22383) | Nature, 2017 | [Code](https://www.ncbi.nlm.nih.gov/search/all/?term=SRP101659) |
| [VDJdb](https://academic.oup.com/nar/article/46/D1/D419/4101254) | Nucl. Acids Res., 2018 | [Code](https://vdjdb.cdr3.net/) |
| [TetTCR-seq](https://www.nature.com/articles/nbt.4282) | Nat. Biotechnol., 2018 | [Code](https://www.ncbi.nlm.nih.gov/gap/) |
| [IEDB](https://academic.oup.com/nar/article/47/D1/D339/5144151) | Nucl. Acids Res., 2019 | [Code](https://www.iedb.org/) |
| [10x](https://www.10xgenomics.com/welcome?closeUrl=%2Flibrary&lastTouchOfferName=A%20new%20way%20of%20exploring%20immunity%3A%20Linking%20highly%20multiplexed%20antigen%20recognition%20to%20immune%20repertoire%20and%20phenotype&lastTouchOfferType=Marketing%20Literature&product=chromium&redirectUrl=%2Flibrary%2Fa14cde) | Tech. Rep., 2019 | [Code](https://www.10xgenomics.com/) |
| [PIRD](https://academic.oup.com/bioinformatics/article/36/3/897/5543102) | Bioinformatics, 2020 | [Code](https://db.cngb.org/pird/) |
| [Heilkkila et al.](https://www.sciencedirect.com/science/article/pii/S016158902030479X) | Mol. Immunol., 2020 | [Code](https://www.ebi.ac.uk/ena/browser/view/PRJEB41936) |
| [NeoTCR](https://academic.oup.com/gpb/article/23/2/qzae010/7600436) | Genomics Proteomics Bioinformatics, 2024 | [Code](http://neotcrdb.bioxai.cn/home) |
| [ImmuneCODE](https://www.frontiersin.org/journals/immunology/articles/10.3389/fimmu.2025.1488851/full) | Front. Immunol., 2025 | [Code](https://clients.adaptivebiotech.com/pub/covid-2020) |
| [TRAIT](https://academic.oup.com/gpb/article/23/3/qzaf033/8116953) | Genomics Proteomics Bioinformatics, 2025 | [Code](https://pgx.zju.edu.cn/traitdb/) |


## Representative AI methods for T-cell antigen identification

### Peptide-MHC I Binding Prediction

| Model | Published in | Datasets used |  N.G. | E.M. | Metrics |  Resources |
|-------|--------------|-----------|-------|--------------|-----------|-----------|
| [NetMHCpan-4.1](https://academic.oup.com/nar/article/48/W1/W449/5837056) | Nucl. Acids Res., 2020 | IEDB |RS | BL |AUROC, PPVn | [Resources](http://www.cbs.dtu.dk/services/NetMHCpan-4.1/) |
| [Anthem](https://academic.oup.com/bib/article/22/5/bbaa415/6102669) | Brief. Bioinform., 2021 | IEDB,EPIMHC,MHCBN,SYFPEITHI |RS | BL |AUROC,Sensitivity,Specificity,Accuracy,MCC | [Resources](https://github.com/17shutao/Anthem) |
| [TransPHLA](https://www.nature.com/articles/s42256-022-00459-7) | Nat. Mach. Intell., 2022 | IEDB,EPIMHC,MHCBN,SYFPEITHI |RS | OR |AUROC,MCC,F1-Score,Accuracy| [Resources](https://github.com/a96123155/TransPHLA-AOMP) |
| [STMHCpan](https://academic.oup.com/bib/article/24/3/bbad164/7147024) | Brief. Bioinform., 2023 | IEDB |RS | OR |AUROC,Recall,Precision,F1-Score,Accuracy | [Resources](https://github.com/Luckysoutheast/STMHCPan) |
| [MixMHCpred2.2](https://www.cell.com/cell-systems/fulltext/S2405-4712(22)00470-7) | Cell Systems, 2023 | Self-curated datasets from multiple public sources |RS | BL |AUROC,PPV | [Resources](https://github.com/GfellerLab/MixMHCpred/releases) |
| [BigMHC](https://www.nature.com/articles/s42256-023-00694-6) | Nat. Mach. Intell., 2023 | IEDB,NEPdb,TESLA,Neopepsee,MANAFEST |RS | OH |AUROC,AUPRC,PPVn | [Resources](https://github.com/KarchinLab/bigmhc) |
| [ImmuneApp](https://www.nature.com/articles/s41467-024-53296-0) | Nat. Commun., 2024 | Self-curated datasets from multiple public sources |RS | BL |AUROC,AUPRC,PPVn | [Resources](https://github.com/bsml320/ImmuneApp) |
| [MixMHCpred3.0](https://link.springer.com/article/10.1186/s13073-025-01450-8) | Genome Med., 2025 | Self-curated datasets from multiple public sources |RS | BL |AUROC,AUPRC | [Resources](https://github.com/GfellerLab/MixMHCpred) |
| [UniPMT](https://www.nature.com/articles/s42256-025-01002-0) | Nat. Mach. Intell., 2025 | IEDB,TESLA,NEPdb,Neopepsee,MANAFEST |RS | LM |AUROC,AUPRC | [Resources](https://github.com/ethanmock/UniPMT) |
| [UnifyImmun](https://www.nature.com/articles/s42256-024-00973-w) | Nat. Mach. Intell., 2025 | IEDB,EPIMHC,MHCBN,SYFPEITHI |RS,NC | OR |AUROC,AUPRC,Accuracy,MCC,F1-Score | [Resources](https://github.com/hliulab/UnifyImmun) |
| [deepAntigen](https://www.nature.com/articles/s41467-025-60461-6) | Nat. Commun., 2025 | IEDB,Sarkizova et al.,Abelin et al.,TESLA,Xu et al. |RS | OH |AUROC,AUPRC,Sensitivity,Specificity,Precision,NPCC | [Resources](https://github.com/JiangBioLab/deepAntigen) |

### Peptide-MHC II Binding Prediction

| Model | Published in | Datasets used |  N.G. | E.M. | Metrics |  Resources |
|-------|--------------|-----------|-------|--------------|-----------|-----------|
| [DeepSeqPanII](https://ieeexplore.ieee.org/abstract/document/9411722) | IEEE/ACM Trans. Comput. Biol. Bioinform., 2021 | IEDB |Not mentioned | OH&BL |AUROC,SRCC | [Resources](https://github.com/pcpLiu/DeepSeqPanII) |
| [DeepMHCII](https://academic.oup.com/bioinformatics/article/38/Supplement_1/i220/6617501) | Bioinformatics, 2022 | IEDB |Not mentioned | OR |AUROC,PCC | [Resources](https://github.com/yourh/DeepMHCII) |
| [MixMHC2pred2.0](https://www.cell.com/immunity/fulltext/S1074-7613(23)00129-2?) | Immunity, 2023 | Self-curated datasets from multiple public sources |RS | OH&BL |AUROC | [Resources](https://github.com/GfellerLab/MixMHC2pred) |
| [NetMHCIIpan4.2](https://www.nature.com/articles/s42003-023-04749-7) | Commun. Biol., 2023 | Self-curated datasets from multiple public sources |RS | BL |AUROC,PPVn | [Resources](https://services.healthtech.dtu.dk/services/NetMHCIIpan-4.2/) |
| [NetMHCIIpan4.3](https://www.science.org/doi/full/10.1126/sciadv.adj6367) | Sci. Adv., 2023 | Self-curated datasets from multiple public sources |RS | BL |AUROC,PPVn | [Resources](https://services.healthtech.dtu.dk/services/NetMHCIIpan-4.3/) |
| [deepAntigen](https://www.nature.com/articles/s41467-025-60461-6) | Nat. Commun., 2025 | Rappazzo et al.,Strazar et al. |RS | OH |AUROC,AUPRC,Sensitivity,Specificity,Precision,NPCC | [Resources](https://github.com/JiangBioLab/deepAntigen) |

### TCR-pMHC Binding Prediction

| Model | Published in | Datasets used |  N.G. | E.M. | Metrics |  Resources |
|-------|--------------|-----------|-------|--------------|-----------|-----------|
| [ERGO-II](https://www.frontiersin.org/journals/immunology/articles/10.3389/fimmu.2021.664514/full) | Front. Immunol., 2021 | McPAS-TCR,VDJdb,Kanakry et al. |SH | OH |AUROC | [Resources](https://github.com/IdoSpringer/ERGO-II) |
| [NetTCR-2.0](https://www.nature.com/articles/s42003-021-02610-3) | Commun. Biol., 2021 | IEDB,VDJdb,10x,MIRA |SH  | PC |AUROC,PPVn | [Resources](https://services.healthtech.dtu.dk/services/NetTCR-2.0/) |
| [ImRex](https://academic.oup.com/bib/article/22/4/bbaa318/6042663) | Brief. Bioinform., 2021 | VDJdb,McPAS-TCR,Dean et al. |BK,SH | PC,BL |AUROC,AUPRC | [Resources](https://github.com/pmoris/ImRex) |
| [DLpTCR](https://academic.oup.com/bib/article/22/6/bbab335/6355415?login=false) | Brief. Bioinform., 2021 | TetTCR-seq,VDJdb,IEDB |BK | OH,PC |Recall,Precision,Accuracy,AUROC | [Resources](https://github.com/jiangBiolab/DLpTCR) |
| [pMTnet](https://www.nature.com/articles/s42256-021-00383-2) | Nat. Mach. Intell., 2021 | PIRD,McPAS-TCR,VDJdb,10x,TetTCR-seq,Chen et al.  |SH | BL |AUROC,AUPRC | [Resources](https://github.com/tianshilu/pMTnet) |
| [DeepTCR](https://www.nature.com/articles/s41467-021-21879-w) | Nat. Commun., 2021 | Dash et al.,10x,Glanville et al.,ImmunoMap,Chan et al. |Not mentioned | OH |AUROC,Recall,Precision,F1-Score | [Resources](https://github.com/sidhomj/DeepTCR) |
| [TITAN](https://academic.oup.com/bioinformatics/article/37/Supplement_1/i237/6319659) | Bioinformatics, 2021 | VDJdb,ImmuneCODE,BindingDB |SH | BL |Accuracy,AUROC | [Resources](https://github.com/PaccMann/TITAN) |
| [PRIME2.0](https://www.cell.com/cell-systems/fulltext/S2405-4712(22)00470-7) | Cell Systems, 2023 | Self-curated datasets from multiple public sources |BK | BL |AUROC,PPV | [Resources](https://github.com/GfellerLab/PRIME) |
| [TEINet](https://academic.oup.com/bib/article/24/2/bbad086/7076118) | Brief. Bioinform., 2023 | VDJdb,McPAS-TCR,Lu et al. |SH,BK | LM |AUROC,Accuracy,Precision,Recall | [Resources](https://github.com/jiangdada1221/TEINet) |
| [PanPep](https://www.nature.com/articles/s42256-023-00619-3) | Nat. Mach. Intell., 2023 | IEDB,VDJdb,PIRD,McPAS-TCR,ImmuneCODE |BK | PC |AUROC,AUPRC | [Resources](https://github.com/bm2-lab/PanPep) |
| [TEIM](https://www.nature.com/articles/s42256-023-00634-4) | Nat. Mach. Intell., 2023 | IEDB,VDJdb,McPAS-TCR,ImmuneCODE |SH | BL |AUROC,MCC,AUPRC | [Resources](https://github.com/pengxingang/TEIM) |
| [PISTE](https://www.nature.com/articles/s42256-024-00901-y) | Nat. Mach. Intell., 2024 | VDJdb,McPAS-TCR,Lu et al.|SH,BK | OR |AUROC,AUPRC,PPVn | [Resources](https://github.com/jychen01/PISTE) |
| [MixTCRpred](https://www.nature.com/articles/s41467-024-47461-8) | Nat. Commun., 2024 | VDJdb,McPAS-TCR,IEDB,10x |SH,BK | OR |AUROC | [Resources](https://github.com/GfellerLab/MixTCRpred) |
| [TPepRet](https://academic.oup.com/bioinformatics/article/41/1/btaf022/7989444) | Bioinformatics, 2025 | IEDB,VDJdb,McPAS-TCR |SH | PC |AUROC,AUPRC | [Resources](https://github.com/CSUBioGroup/TPepRet) |
| [UniPMT](https://www.nature.com/articles/s42256-025-01002-0) | Nat. Mach. Intell., 2025 | IEDB,TetTCR-seq,VDJdb,McPAS-TCR,PIRD |SH | LM |AUROC,AUPRC | [Resources](https://github.com/ethanmock/UniPMT) |
| [UnifyImmun](https://www.nature.com/articles/s42256-024-00973-w) | Nat. Mach. Intell., 2025 | TetTCR-seq,VDJdb,IEDB,PIRD,Heilkkila et al.,10x,ImmuneCODE,BindingDB |SH,BK | OR |AUROC,AUPRC,Accuracy,MCC,F1-Score | [Resources](https://github.com/hliulab/UnifyImmun) |
| [TCRBagger](https://www.cell.com/cell-systems/abstract/S2405-4712(25)00236-4) | Cell System, 2025 | IEDB,VDJdb,McPAS-TCR,PIRD |BK | PC |AUROC,AUPRC | [Resources](https://github.com/bm2-lab/TCRBagger) |
| [deepAntigen](https://www.nature.com/articles/s41467-025-60461-6) | Nat. Commun., 2025 | IEDB,VDJdb,McPAS-TCR,PIRD,ImmuneCODE,NeoTCR  |BK | OH |AUROC,AUPRC,Sensitivity,Specificity,Precision,NPCC | [Resources](https://github.com/JiangBioLab/deepAntigen) |

N.G.=Negatives generation,
E.M.=Embedding method,
RS=Negatives by randomly sampling,
NC=Negatives from non-cognate  MHC alleles,
SH = Negatives by shuffling, BK = Negatives from background data.
OH = one-hot embedding, 
OR = ordinal embedding, 
PC = physicochemical property (or Atchley factors)-based embedding, 
KM=K-mer feature-based embedding, 
BL=BLOSUM-based embedding, 
LM=language model-based embedding.

        
## Comprehensive Evaluation of 18 TCR-pMHC Binding Predictors

To provide a much-needed, unified assessment and establish a reproducible benchmark for the field, we conduct a comprehensive, head-to-head evaluation of 18 state-of-the-art TCR-pMHC binding prediction methods mentioned above. In order to make a fair comparison, all models are re-implemented and evaluated using a consistent framework, including unified data preprocessing (one-hot encoding for sequences and categorical attributes like V/J genes and MHC), negative sample generation, and training parameters (i.e., 40 epochs, batch size of 64, Adam optimizer with learning rate 0.0002 and weight decay 1e-5 on McPAS-TCR dataset; 80 epochs, batch size of 64, Adam optimizer with learning rate 0.0001 and weight decay 1e-5 on IEDB and VDJdb datasets). As some original implementations are not fully open-sourced or are based on different deep learning frameworks (e.g., Keras), we re-implement all models in PyTorch; while we strive to preserve the original performance, we cannot guarantee 100\% replication of the original results. Evaluations are performed on three major public databases: the [IEDB](https://academic.oup.com/nar/article/47/D1/D339/5144151), [McPAS-TCR](https://academic.oup.com/bioinformatics/article/33/18/2924/3803440), and [VDJdb](https://academic.oup.com/nar/article/46/D1/D419/4101254). The results are shown below.
<img width="2351" height="2915" alt="figure_07" src="https://github.com/user-attachments/assets/7495a55a-8524-4274-83fa-4db3a719df74" />


To assess the clinical utility and robustness of the benchmarked methods, we conduct a critical out-of-distribution (OOD) generalization test on two independent Unseen Epitope Variant Datasets (Dataset I and Dataset II). These datasets, referenced from the stringent [ePytope-TCR benchmark](https://www.cell.com/cell-genomics/fulltext/S2666-979X(25)00202-2), are specifically designed to challenge models with novel peptide sequences, mirroring the difficulties encountered in real-world contexts where neoantigens frequently arise. The comprehensive performance evaluation is presented as follows.


![figure_08](https://github.com/user-attachments/assets/055a15b4-918d-4e8a-b5bf-c8d2c4483731)

### Code Introduction

- **data_preprocess.py**: preprocess the raw downloaded dataset of IEDB, McPAS, and VDJdb.
- **config.py**: the file including the necessary settings.
- **model.py**: the file including all models.
- **main.py**: the main file to run all models.
- **filter_data.py**: filter the dataset to ensure the independence of two unseen OOD datasets.
- **test_new_data.py**: to evaluate the trained model on the two unseen OOD datasets.
- **data**: the folder including the raw downloaded datasets: IEDB, McPAS, and VDJdb. The size of the raw IEDB dataset is too large (>25MB) to upload to GitHub. You can download it by following the instructions in the guide file data_download.txt located in the data folder.   
- **unseen_data**: the folder including the unseen dataset downloaded from [ePytope-TCR benchmark](https://zenodo.org/records/15025579).
## License 

MIT License

