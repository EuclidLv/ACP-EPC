# ACP-EPC

**ACP-EPC** is an anticancer peptide predictor that integrates **ESM-2 embeddings** and **traditional physicochemical descriptors** using deep learning.  
It applies **Cross-Attention fusion** to combine structural and biochemical features, achieving state-of-the-art performance on multiple benchmark datasets.  
A user-friendly **web server** is also provided for public access.  

---

##  Overview

<img width="865" height="880" alt="image" src="https://github.com/user-attachments/assets/de3bef5b-3962-45ea-b9f4-c0aa8d6ac138" />


**Figure 1.** Overview of ACP-EPC.  
(A) Dataset collection and construction.  
(B) Model architecture.  
(C) Web server development.  

---

##  Features
- **Hybrid feature extraction**: Combines contextual embeddings from **ESM-2** with handcrafted physicochemical descriptors.  
- **Cross-Attention fusion**: Dynamically integrates structural and biochemical information.  
- **Robust evaluation**: Validated on two datasets (ACP135, ACP194, ACP99).  
- **User-friendly web server**: No installation required, supports FASTA batch submission.  
- **Open-source reproducibility**: Trained weights, scaler, datasets, and evaluation metrics are provided.  

---

##  Web Server

The ACP-EPC web server is freely available at:  
 [http://121.36.197.223:45131/](http://121.36.197.223:45131/)

**Notes:**  
1. The web server only accepts sequences with length **< 1024**.  
2. For questions, please contact us at: *your_email@domain.com*  

---

##  Performance

### Overall Metrics

| Dataset   | Accuracy | Specificity | Sensitivity | MCC   | 
|-----------|----------|-------------|-------------|-------|
| ACP135    | 0.935    | 0.964       | 0.859       | 0.835 |
| ACP194    | 0.984    | 0.987       | 0.980       | 0.993 |

ACP-EPC consistently outperforms existing baselines across two datasets.  

<img width="867" height="780" alt="image" src="https://github.com/user-attachments/assets/aaa0680a-33c7-4950-9b4d-3f4527b353f9" />


**Figure 4. Performance of ACP-EPC on ACP135 and ACP99.**  
(A) KDE of predicted probability *p(y=ACP)* on ACP135. Blue means ACPs; orange means Non-ACPs. ACPs cluster near 1.0 and Non-ACPs near 0.0 with limited overlap, indicating strong separability.  
(B) KDE of *p(y=ACP)* on ACP99.  
(C) ROC for ACP135 (blue) and ACP99 (orange); axes: TPR vs FPR. The dashed diagonal denotes random. Reported AUCs: **0.9682** (ACP135) and **0.9933** (ACP99).  
(D) PR for the positive class using the same predictions; axes: precision vs recall. Reported AUC(PR): **0.9496** (ACP135) and **0.9869** (ACP99), showing high precision sustained across broad recall, especially on ACP99.  

---

###  Comparative Performance

**Table 4. Performance comparison of ACP-EPC with existing methods on ACP135.**

| Model      | ACC   | Sn    | Sp    | MCC   |
|------------|-------|-------|-------|-------|
| ACPred-BMF | 0.669 | 0.838 | 0.429 | 0.296 |
| ACP-MHCNN  | 0.590 | 0.783 | 0.338 | 0.136 |
| ACPred     | 0.857 | 0.914 | 0.719 | 0.648 |
| iDACP      | 0.879 | 0.861 | **0.975** | 0.687 |
| mACPpred   | 0.871 | 0.879 | 0.838 | 0.659 |
| ACP-ML     | 0.909 | **0.939** | 0.830 | 0.770 |
| **ACP-EPC**| **0.935** | 0.859 | 0.964 | **0.835** |

---

**Table 5. Performance comparison of ACP-EPC with existing methods on ACP99.**

| Model      | ACC   | Sn    | Sp    | MCC   |
|------------|-------|-------|-------|-------|
| mACPpred   | 0.895 | 0.882 | 0.918 | 0.777 |
| ACPred     | 0.864 | 0.896 | 0.814 | 0.714 |
| ACP-MHCNN  | 0.699 | 0.738 | 0.625 | 0.354 |
| ACPred-BMF | 0.758 | 0.920 | 0.629 | 0.561 |
| ACP-ML     | 0.926 | 0.937 | 0.908 | 0.843 |
| **ACP-EPC**| **0.984** | **0.980** | **0.987** | **0.993** |

---

##  Repository Structure

```bash
ACP-EPC/
│── ACP-EPC.py        # Main predictor script
│── config.json       # Python version & package dependencies
│── Data/             # Trained weights & datasets
│   ├── best_model_focal.pth
│   ├── scaler.npz
│   ├── test_metrics_focal.csv
│   ├── training.fasta
│   ├── testing99.fasta
│   └── testing135.fasta
