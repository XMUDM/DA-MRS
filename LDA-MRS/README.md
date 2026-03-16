# LDA-MRS

This repository contains our implementations for "Lightweight Denoising and Aligning Multi-modal Recommender System". 



### Folder structure

- data: The public links to download the datasets and dataset preprocessing methods. 
- log_files: The log files to quickly reproduce the reported results. 
- src_LDAMRS: The implementation of LDA-MRS on Amazon Baby, Sports and Clothing. 



### Dependencies

- OS: Ubuntu 20.04.6 LTS
- numpy==1.23.5
- pandas==1.5.3
- python==3.10.9
- scipy==1.10.0
- torch==2.0.0
- pyyaml==6.0
- pyg==2.3.0
- networkx==2.8.4
- tqdm==4.65.0
- lmdb==1.4.0



### Main Experiment

1. Step 1: Download and pre-process the datasets to folder './data' following the instruction in './data/Readme.md'.  

2. Step 2: Run the following command to construct item-item behavior graph. 

   ```python
   # construct item-item behavior graph. 
   
   python -u build_iib_graph.py --dataset=baby --topk=2
   ```

   There are two parameters:

   1. dataset: str type, allowed values are baby, sports, clothing, and tiktok.
   2. topk: int type, parameter for pruning the Item-item behavior graph.

   

3. Step 3: Specify configurations

   1. Enter to the './src_LDAMRS' folder

      ```bash
      # enter to the src folder
      
      cd src_LDAMRS
      ```

   2. Specify dataset-specific configurations

      ```bash
      vim configs/dataset/xx.yaml
      ```

   3. Specify model-specific configurations

      ```bash
      vim configs/model/xx.yaml
      ```

   4. Specify the overall configurations

      ```bash
      vim configs/overall.yaml
      ```
   5. Rename the model filename
      ```bash
      # when run the lightgcn-ldarms-s
      cp models/lightgcn-ldarms-s.py models/lightgcn.py

      # when run the lightgcn-ldarms-d
      cp models/lightgcn-ldarms-d.py models/lightgcn.py
      ```

4. Step 4: Run the following command to train and evaluate the model. 

   ```python
   # run the code 
   
   python -u main.py --model=LightGCN --dataset=baby --gpu_id=0
   ```

   There are three parameters: 

   1. model: str type, the name of the **backbone model**, such as LightGCN. 

   2. dataset: str type, the name of the dataset, such as baby, sports, clothing.

   3. gpu_id: str type, the specified GPU. 


### Acknowledgement

The structure of this code is based on [MMRec](https://github.com/enoche/MMRec). Thanks for their work.
