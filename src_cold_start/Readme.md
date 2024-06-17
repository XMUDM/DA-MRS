1. Step 1: Run the following commend to construct item-item behavior graph. 

   ```python
   # construct item-item behavior graph. 
   
   python -u build_iib_graph.py --dataset=baby --topk=2
   ```

   There are two parameters:

   1. dataset: str type, allowed values are baby, sports, clothing, and tiktok.

   2. topk: int type, parameter for pruning the Item-item behavior graph.

      

2. Step 2: Specify configurations

   1. Go to the src_cold_start folder

      ```bash
      # go to the src folder
      cd src_cold_start
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

   5. Differentiating user groups

      **In utils/dataset.py , lines 70-74**, we can differentiat user groups. 

   ```python
    a = self.df['userID'].value_counts()
    # set the active user group
    warm = a[a>=50].index.tolist()
   
    # set the new user group 
    cold = a[a==5].index.tolist()
    dfs.append(dfs[2][dfs[2]['userID'].isin(warm)])
    dfs.append(dfs[2][dfs[2]['userID'].isin(cold)])
   ```

3. Step 3: Run the following commend to train and evaluate the model. 

   ```python
   # run the code 
   python -u main.py --model=LightGCN --dataset=baby --gpu_id=0
   ```

   There are three parameters: 

   1. model: str type, the name of the backbone model, such as LightGCN and MF.

   2. dataset: str type, the name of the dataset, such as baby, sports, clothing, and tiktok.

   3. gpu_id: str type, the specified GPU. 

      

4. **We provide the logs of DA-MRS+LightGCN and LightGCN in the log/ folder for reference.** 

   - DA-MRS+LightGCN-baby-Dec-10-2023-17-46-27.log: The results of active user group and less active user group. 
   - DA-MRS+LightGCN-baby-Dec-10-2023-18-18-27.log: The results of active user group and new user group. 
   - LightGCN-baby-Dec-10-2023-16-57-36.log: The results of active user group and new user group. 
   - LightGCN-baby-Dec-10-2023-18-20-02.log: The results of active user group and less active user group. 