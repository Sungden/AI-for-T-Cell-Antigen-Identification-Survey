# AI for T-Cell Antigen Identification: A Comprehensive Survey
AI-Driven Computational Methods and Data Resources for T-Cell Antigen Identification

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
| [10x](https://www.nature.com/articles/s41587-019-0322-9) | Tech. Rep., 2019 | [Code](https://www.10xgenomics.com/) |
| [PIRD](https://academic.oup.com/bioinformatics/article/36/3/897/5543102) | Bioinformatics, 2020 | [Code](https://db.cngb.org/pird/) |
| [Heilkkila et al.](https://www.sciencedirect.com/science/article/pii/S016158902030479X) | Mol. Immunol., 2020 | [Code](https://www.ebi.ac.uk/ena/browser/view/PRJEB41936) |
| [NeoTCR](https://academic.oup.com/gpb/article/23/2/qzae010/7600436) | Genomics Proteomics Bioinformatics, 2024 | [Code](http://neotcrdb.bioxai.cn/home) |
| [ImmuneCODE](https://www.frontiersin.org/journals/immunology/articles/10.3389/fimmu.2025.1488851/full) | Front. Immunol., 2025 | [Code](https://clients.adaptivebiotech.com/pub/covid-2020) |
| [TRAIT](https://academic.oup.com/gpb/article/23/3/qzaf033/8116953) | Genomics Proteomics Bioinformatics, 2025 | [Code](https://pgx.zju.edu.cn/traitdb/) |


## Representative AI methods for T-cell antigen identification

### Peptide-MHC I Binding Prediction

| Model | Published in | Datasets used |  N.G. | E.M. | Metrics |  Resources |
|-------|--------------|-----------|-------|--------------|-----------|-----------|
| [NetMHCpan-4.1](https://link.springer.com/article/10.1007/S002510050595) | Nucl. Acids Res., 2020 | IEDB |RS | BL |AUROC, PPVn | [Resources](http://www.cbs.dtu.dk/services/NetMHCpan-4.1/) |
| [Anthem](https://link.springer.com/article/10.1007/S002510050595) | Brief. Bioinform., 2021 | IEDB,EPIMHC,MHCBN,SYFPEITHI |RS | BL |AUROC,Sensitivity,Specificity,Accuracy,MCC | [Resources](http://www.cbs.dtu.dk/services/NetMHCpan-4.1/) |
| [TransPHLA](https://link.springer.com/article/10.1007/S002510050595) | Nat. Mach. Intell., 2022 | IEDB,EPIMHC,MHCBN,SYFPEITHI |RS | OR |AUROC,MCC,F1-Score,Accuracy| [Resources](http://www.cbs.dtu.dk/services/NetMHCpan-4.1/) |
| [STMHCpan](https://link.springer.com/article/10.1007/S002510050595) | Brief. Bioinform., 2023 | IEDB |RS | OR |AUROC,Recall,Precision,F1-Score,Accuracy | [Resources](http://www.cbs.dtu.dk/services/NetMHCpan-4.1/) |
| [MixMHCpred2.2](https://link.springer.com/article/10.1007/S002510050595) | Cell Systems, 2023 | Self-curated datasets from multiple public sources |RS | BL |AUROC,PPV | [Resources](http://www.cbs.dtu.dk/services/NetMHCpan-4.1/) |
| [BigMHC](https://link.springer.com/article/10.1007/S002510050595) | Nat. Mach. Intell., 2023 | IEDB,NEPdb,TESLA,Neopepsee,MANAFEST |RS | OH |AUROC,AUPRC,PPVn | [Resources](http://www.cbs.dtu.dk/services/NetMHCpan-4.1/) |
| [ImmuneApp](https://link.springer.com/article/10.1007/S002510050595) | Nat. Commun., 2024 | Self-curated datasets from multiple public sources |RS | BL |AUROC,AUPRC,PPVn | [Resources](http://www.cbs.dtu.dk/services/NetMHCpan-4.1/) |
| [MixMHCpred3.0](https://link.springer.com/article/10.1007/S002510050595) | Genome Med., 2025 | Self-curated datasets from multiple public sources |RS | BL |AUROC,AUPRC | [Resources](http://www.cbs.dtu.dk/services/NetMHCpan-4.1/) |
| [UniPMT](https://link.springer.com/article/10.1007/S002510050595) | Nat. Mach. Intell., 2025 | IEDB,TESLA,NEPdb,Neopepsee,MANAFEST |RS | LM |AUROC,AUPRC | [Resources](http://www.cbs.dtu.dk/services/NetMHCpan-4.1/) |
| [UnifyImmun](https://link.springer.com/article/10.1007/S002510050595) | Nat. Mach. Intell., 2025 | IEDB,EPIMHC,MHCBN,SYFPEITHI |RS,NC | OR |AUROC,AUPRC,Accuracy,MCC,F1-Score | [Resources](http://www.cbs.dtu.dk/services/NetMHCpan-4.1/) |
| [deepAntigen](https://link.springer.com/article/10.1007/S002510050595) | Nat. Commun., 2025 | IEDB,Sarkizova et al.,Abelin et al.,TESLA,Xu et al. |RS | OH |AUROC,AUPRC,Sensitivity,Specificity,Precision,NPCC | [Resources](http://www.cbs.dtu.dk/services/NetMHCpan-4.1/) |

### Peptide-MHC II Binding Prediction

| Model | Published in | Datasets used |  N.G. | E.M. | Metrics |  Resources |
|-------|--------------|-----------|-------|--------------|-----------|-----------|
| [DeepSeqPanII](https://link.springer.com/article/10.1007/S002510050595) | IEEE/ACM Trans. Comput. Biol. Bioinform., 2021 | IEDB |Not mentioned | OH&BL |AUROC,SRCC | [Resources](http://www.cbs.dtu.dk/services/NetMHCpan-4.1/) |
| [DeepMHCII](https://link.springer.com/article/10.1007/S002510050595) | Bioinformatics, 2022 | IEDB |Not mentioned | OR |AUROC,PCC | [Resources](http://www.cbs.dtu.dk/services/NetMHCpan-4.1/) |
| [MixMHC2pred2.0](https://link.springer.com/article/10.1007/S002510050595) | Immunity, 2023 | Self-curated datasets from multiple public sources |RS | OH&BL |AUROC | [Resources](http://www.cbs.dtu.dk/services/NetMHCpan-4.1/) |
| [NetMHCIIpan4.2](https://link.springer.com/article/10.1007/S002510050595) | Commun. Biol., 2023 | Self-curated datasets from multiple public sources |RS | BL |AUROC,PPVn | [Resources](http://www.cbs.dtu.dk/services/NetMHCpan-4.1/) |
| [NetMHCIIpan4.3](https://link.springer.com/article/10.1007/S002510050595) | Sci. Adv., 2023 | Self-curated datasets from multiple public sources |RS | BL |AUROC,PPVn | [Resources](http://www.cbs.dtu.dk/services/NetMHCpan-4.1/) |
| [deepAntigen](https://link.springer.com/article/10.1007/S002510050595) | Nat. Commun., 2025 | Rappazzo et al.,Strazar et al. |RS | OH |AUROC,AUPRC,Sensitivity,Specificity,Precision,NPCC | [Resources](http://www.cbs.dtu.dk/services/NetMHCpan-4.1/) |

### TCR-pMHC Binding Prediction

| Model | Published in | Datasets used |  N.G. | E.M. | Metrics |  Resources |
|-------|--------------|-----------|-------|--------------|-----------|-----------|
| [ERGO-II](https://link.springer.com/article/10.1007/S002510050595) | Front. Immunol., 2021 | McPAS-TCR,VDJdb,Kanakry et al. |SH | OH |AUROC | [Resources](https://github.com/IdoSpringer/ERGO-II) |
| [NetTCR-2.0](https://link.springer.com/article/10.1007/S002510050595) | Commun. Biol., 2021 | IEDB,VDJdb,10x,MIRA |SH  | PC |AUROC,PPVn | [Resources](https://services.healthtech.dtu.dk/services/NetTCR-2.0/) |
| [ImRex](https://link.springer.com/article/10.1007/S002510050595) | Brief. Bioinform., 2021 | VDJdb,McPAS-TCR,Dean et al. |BK,SH | PC,BL |AUROC,AUPRC | [Resources](https://github.com/pmoris/ImRex) |
| [DLpTCR](https://link.springer.com/article/10.1007/S002510050595) | Brief. Bioinform., 2021 | TetTCR-seq,VDJdb,IEDB |BK | OH,PC |Recall,Precision,Accuracy,AUROC | [Resources](https://github.com/jiangBiolab/DLpTCR) |
| [pMTnet](https://link.springer.com/article/10.1007/S002510050595) | Nat. Mach. Intell., 2021 | PIRD,McPAS-TCR,VDJdb,10x,TetTCR-seq,Chen et al.  |SH | BL |AUROC,AUPRC | [Resources](https://github.com/tianshilu/pMTnet) |
| [DeepTCR](https://link.springer.com/article/10.1007/S002510050595) | Nat. Commun., 2021 | Dash et al.,10x,Glanville et al.,ImmunoMap,Chan et al. |Not mentioned | OH |AUROC,Recall,Precision,F1-Score | [Resources](https://github.com/sidhomj/DeepTCR) |
| [TITAN](https://link.springer.com/article/10.1007/S002510050595) | Bioinformatics, 2021 | VDJdb,ImmuneCODE,BindingDB |SH | BL |Accuracy,AUROC | [Resources](https://github.com/PaccMann/TITAN) |
| [PRIME2.0](https://link.springer.com/article/10.1007/S002510050595) | Cell Systems, 2023 | Self-curated datasets from multiple public sources |BK | BL |AUROC,PPV | [Resources](https://github.com/GfellerLab/PRIME) |
| [TEINet](https://link.springer.com/article/10.1007/S002510050595) | Brief. Bioinform., 2023 | VDJdb,McPAS-TCR,Lu et al. |SH,BK | LM |AUROC,Accuracy,Precision,Recall | [Resources](https://github.com/jiangdada1221/TEINet) |
| [PanPep](https://link.springer.com/article/10.1007/S002510050595) | Nat. Mach. Intell., 2023 | IEDB,VDJdb,PIRD,McPAS-TCR,ImmuneCODE |BK | PC |AUROC,AUPRC | [Resources](https://github.com/bm2-lab/PanPep) |
| [TEIM](https://link.springer.com/article/10.1007/S002510050595) | Nat. Mach. Intell., 2023 | IEDB,VDJdb,McPAS-TCR,ImmuneCODE |SH | BL |AUROC,MCC,AUPRC | [Resources](https://github.com/pengxingang/TEIM) |
| [PISTE](https://link.springer.com/article/10.1007/S002510050595) | Nat. Mach. Intell., 2024 | VDJdb,McPAS-TCR,Lu et al.|SH,BK | OR |AUROC,AUPRC,PPVn | [Resources](https://github.com/jychen01/PISTE) |
| [MixTCRpred](https://link.springer.com/article/10.1007/S002510050595) | Nat. Commun., 2024 | VDJdb,McPAS-TCR,IEDB,10x |SH,BK | OR |AUROC | [Resources](https://github.com/GfellerLab/MixTCRpred) |
| [TPepRet](https://link.springer.com/article/10.1007/S002510050595) | Bioinformatics, 2025 | IEDB,VDJdb,McPAS-TCR |SH | PC |AUROC,AUPRC | [Resources](https://github.com/CSUBioGroup/TPepRet) |
| [UniPMT](https://link.springer.com/article/10.1007/S002510050595) | Nat. Mach. Intell., 2025 | IEDB,TetTCR-seq,VDJdb,McPAS-TCR,PIRD |SH | LM |AUROC,AUPRC | [Resources](https://github.com/ethanmock/UniPMT) |
| [UnifyImmun](https://link.springer.com/article/10.1007/S002510050595) | Nat. Mach. Intell., 2025 | TetTCR-seq,VDJdb,IEDB,PIRD,Heilkkila et al.,10x,ImmuneCODE,BindingDB |SH,BK | OR |AUROC,AUPRC,Accuracy,MCC,F1-Score | [Resources](https://github.com/hliulab/UnifyImmun) |
| [TCRBagger](https://link.springer.com/article/10.1007/S002510050595) | Cell System, 2025 | IEDB,VDJdb,McPAS-TCR,PIRD |BK | PC |AUROC,AUPRC | [Resources](https://github.com/bm2-lab/TCRBagger) |
| [deepAntigen](https://link.springer.com/article/10.1007/S002510050595) | Nat. Commun., 2025 | IEDB,VDJdb,McPAS-TCR,PIRD,ImmuneCODE,NeoTCR  |BK | OH |AUROC,AUPRC,Sensitivity,Specificity,Precision,NPCC | [Resources](https://github.com/JiangBioLab/deepAntigen) |


## Description

This repository provides a comprehensive overview of **Large Language Models (LLMs)** and their applications in **protein understanding** and **prediction**. The methods discussed span various domains within protein science, including protein sequence modeling, engineering, and interaction prediction.

### References

- **Nature Methods, 2019**: Unified rational protein engineering with sequence-based deep representation learning
- **ICLR, 2019**: Learning protein sequence embeddings using information from structure
- **NAR Genomics and Bioinformatics, 2020**: Mutation effect estimation on protein–protein interactions using deep contextualized representation learning
- **IEEE Transactions on Pattern Analysis and Machine Intelligence, 2021**: Prottrans: Toward understanding the language of life through self-supervised learning

## License

MIT License

