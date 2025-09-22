# Face ID Retrieval System for NADRA

## Project Overview
This project implements a **Face ID system** capable of retrieving a person's CNIC and other information using only their face image. It is designed to handle **one-shot recognition** scenarios, where only a single reference image per person is available. The system is trained and tested using the **LFW dataset**.

## Features
- **Siamese Network** for learning embeddings from matching and non-matching image pairs.
- **One-shot recognition** using embeddings.
- **Inference options:**
  - **KNN-based retrieval**
  - **FAISS-based vector database retrieval**
  - Ability to switch between custom-trained and pretrained backbone.
- **Evaluation:**  
  - Accuracy and **ROC-AUC** on pairs  
  - Top-1 accuracy for single-shot retrieval

## Dataset
- LFW dataset downloaded from [Kaggle](https://www.kaggle.com/jessicali9530/lfw-dataset)
- Dataset structure: images organized by person name.

## How to Use
1. Open the notebook in **Google Colab** or **Jupyter Notebook**.
2. Install necessary packages: `torch`, `torchvision`, `faiss-cpu`, `kagglehub`.
3. Run all cells sequentially to:
   - Load and preprocess the dataset
   - Generate matching and non-matching pairs
   - Train the Siamese network
   - Build embeddings database
   - Perform single-shot recognition using KNN or FAISS
   - Evaluate model performance (**Accuracy, ROC-AUC, Top-1 retrieval accuracy**)
4. Test with any input image to retrieve the person’s ID.


## Notes
- The system demonstrates **single-shot face recognition** and can scale to large datasets.
- FAISS is recommended for **faster large-scale retrieval** compared to KNN.
