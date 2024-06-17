### Data
Download from Google Drive: [Baby/Sports/Clothing](https://drive.google.com/drive/folders/13cBy1EA_saTUuXxVllKgtfci2A09jyaG?usp=sharing)
The data already contains text and image features extracted from Sentence-Transformers and CNN, which is provided by [MMRec](https://github.com/enoche/MMRec).
Please move your downloaded data into the folder for model training.



Download from Google Drive: [TikTok](https://drive.google.com/drive/folders/1hLvoS7F0R_K0HBixuS_OVXw_WbBxnshF?usp=share_link)
The data already contains text, image and audio features extracted from pre-trained models, which is provided by [MMSSL](https://github.com/HKUDS/MMSSL).
Please move your downloaded data into the folder. Please run the following code to process the TikTok dataset. 

```python
python -u process.py
```

To obtain the average item similarity based on each modality (e.g., textual content, visual content, audio content, etc.) of these datasets, you can run calculate_avg_similar.py 

```python
python -u calculate_avg_similar.py
```