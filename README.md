# Detecting COVID-19 from Chest X-Ray Images Using PyTorch in Google Colab

This project implements a deep learning model to classify chest X-ray images into three classes: Normal, Viral Pneumonia, and COVID-19.

## Description

The goal of this project is to classify chest X-ray images into three categories: Normal, Viral Pneumonia, and COVID-19. The dataset used in this project is a comprehensive collection of images curated by a collaborative team of researchers and medical doctors from Qatar University, the University of Dhaka, and other international partners. You can find the dataset on Kaggle [here](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database).

The dataset contains a total of 21,165 chest X-ray images, including a significant increase in COVID-19 positive cases over time:

- 10,192 normal images
- 6,012 lung opacity (non-COVID lung infection) images
- 1,345 viral pneumonia images
- 3,616 COVID-19 images

Originally, the dataset started with only 219 COVID-19 images, but it has been updated to include more cases to address the class imbalance.

## Approach

The approach taken in this project involves several key steps to ensure the model is trained effectively:

1. **Dataset Splitting**: The dataset is split to reserve a validation set of 30 images per class (90 images total). This allows for measuring the model's accuracy during training.

2. **Model Selection**: 
   - A ResNet18 network is used, with weights pre-trained on ImageNet. Only the final classification layer is adjusted to predict between the 3 classes: Normal, Viral Pneumonia, and COVID-19.
   - All model parameters are fine-tuned during training to optimize performance.

3. **Training Conditions**:
   - **Optimizer**: Adam optimizer with a learning rate of 3e-5.
   - **Loss Function**: PyTorch's cross-entropy loss function is used for multi-class classification.
   - **Batch Size**: A batch size of 6 is used for training.
   - **Image Processing**: Images are resized to (224, 224) and normalized according to the model requirements.

4. **Model Performance**:
   - The model converges in less than one epoch and achieves an accuracy of over 98% on the validation images.

## How to Use the Notebook

This guide will help you set up and run the notebook for your project.

### Step 1: Download the Dataset
- Download the [dataset](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database) from Kaggle. Ensure that you have a Kaggle account and are logged in to access the dataset.

### Step 2: Upload the Dataset to Google Drive
- After downloading, upload the dataset to your Google Drive. This makes it easily accessible when running the notebook on Google Colab.

### Step 3: Run the Notebook on Google Colab
- Open the notebook in Google Colab by clicking the link or uploading it directly to Colab.
- Locate the section in the notebook where the dataset is loaded from Google Drive. Update the link to point to your own Google Drive path if necessary. Replace any existing links with the link to your uploaded dataset to ensure proper loading.

### Step 4: Execute the Cells
- Run each cell sequentially to execute the code. This will set up the environment, load the dataset, train the model, and visualize the results.

### Tips for Running the Notebook
- Ensure you have a stable internet connection, as the notebook will need to access files from Google Drive and may require additional library installations.
- Adjust any hyperparameters or file paths according to your specific needs, especially if you encounter errors related to data loading or file paths.

### Troubleshooting
- **Dataset Not Found**: Double-check your Google Drive link and ensure the path specified in the notebook matches the folder structure of your dataset.
- **Permission Errors**: If you encounter permission errors when accessing Google Drive, ensure that you have authenticated your Google account properly and granted the necessary permissions.

## Acknowledgments
This dataset was developed through a collaborative effort by researchers from Qatar University, the University of Dhaka, and their collaborators from Pakistan and Malaysia, along with medical doctors. We thank them for their work in providing this valuable resource.

By following this guide, you should be able to run the notebook successfully and utilize the deep learning model to classify chest X-ray images effectively.
