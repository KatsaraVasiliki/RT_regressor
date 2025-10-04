# Retention Time Prediction with Deep Learning and Transfer Learning

This repository contains the code and resources developed for the thesis:  
**Prediction of the retention time of natural product metabolites using transfer learning strategies (http://dx.doi.org/10.26265/polynoe-7669)**.  

The goal of this work is to **predict retention times (RTs) in chromatography** using **deep neural networks (DNNs)**, with a focus on **transfer learning** to adapt models trained on synthetic compounds to natural products.  

## Research Affiliation  

This research was conducted within the **Biomedical Engineering Departments** of:  

- **Universidad San Pablo CEU (Madrid, Spain)**  
- **University of West Attica (Athens, Greece)**


And under the supervision of:  

- **Abraham Otero Quintana**  
- **Guillermo Ramajo Fernández**  
- **Minos-Timotheos Matsoukas** 


---
##  Challenges and Problem Statement  

Deep neural networks (DNNs) generally require **large, labeled datasets** to achieve high performance.  
In chromatography, this poses a challenge:  

- **SMRT dataset**: Large (~80,000 compounds) but mostly **synthetic compounds**, which have different chemical properties than natural products.  
- **Natural product RT datasets**: Much **smaller** (dozens to hundreds of compounds), making them less suitable for training DNNs directly.  

As a result, models trained on SMRT often fail to generalize well to natural products, while models trained directly on natural product datasets suffer from **data scarcity** and limited predictive power.  

**Transfer learning** provides a solution:  
- First, train a model on the large SMRT dataset to capture general molecular features influencing RT.  
- Then, fine-tune this pre-trained model on smaller natural product datasets (RepoRT), enabling better predictions despite limited data.  

### Project Goal  
This project aims to:  
1. Address the limitations of SMRT (synthetic bias) and small natural product datasets.  
2. Use **transfer learning** to improve retention time prediction for natural compounds.  
3. Compare transfer learning with direct training approaches, evaluating their performance across multiple error metrics.  


---

## Research Summary  

- **Datasets**:  
  - **SMRT dataset** (~80,000 synthetic compounds, HPLC-MS) used to train the base DNN model.  
  - **RepoRT dataset** (natural products, 373 experiments) used for fine-tuning and evaluation.  

- **Approaches explored**:  
  - **Direct Training**: Train a new DNN for each RepoRT experiment.  
  - **Transfer Learning**: Fine-tune the SMRT-trained DNN on RepoRT experiments using a two-stage strategy:  
     - Freeze upper layers → train lower layers.  
     - Unfreeze all layers → fine-tune with reduced learning rate.  

- **Model Features**:  
  - Fingerprints, descriptors, and combined features generated with **alvadesc**.  
  - Both **sequential** and **functional (deep & wide)** DNN architectures were tested.  

- **Evaluation Metrics**:  
  - Mean Absolute Error (**MAE**)  
  - Median Absolute Error (**MedAE**)  
  - Mean Absolute Percentage Error (**MAPE**)  

- **Key Findings**:  
  - Transfer learning outperformed direct training in terms of MAE and MedAE.  
  - Direct training performed better for MAPE.  
  - After removing outliers, transfer learning consistently performed better across all metrics.  
  - Transfer learning also proved more computationally efficient, requiring fewer hyperparameter optimizations.  

For more details, see the master thesis: http://dx.doi.org/10.26265/polynoe-7669

---
## Citation

The model trained with the SMRT dataset is inspired by
> García, C.A., Gil-de-la-Fuente, A., Barbas, C. et al. Probabilistic metabolite annotation using retention time prediction and meta-learned projections. J Cheminform 14, 33 (2022). https://doi.org/10.1186/s13321-022-00613-8. 

---


## Repository Structure  

- **`main.py`**  
  - Defines the main DNN architecture.  
  - Trains the model on the **SMRT dataset** using features from alvadesc.  
  - Supports fingerprints, descriptors, and combined features.  

- **`Transfer_learning_2.py`**  
  - Implements the **transfer learning process**.  
  - Loads the SMRT-trained model and fine-tunes it on RepoRT datasets.  

- **`NoSMRT_NoTfl.py`**  
  - Implements **direct training** (new model trained from scratch) for each RepoRT dataset.  
  - Allows direct comparison between transfer learning and direct training approaches.  



---

## Environment Setup  

The repository includes an exported environment file: `environment.yml`  

Export, create and activate the environment with:  
```bash
conda env export > environment.yml
conda env create -f environment.yml --name cmmrt_env
conda activate cmmrt_env


