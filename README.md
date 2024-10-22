# Melanoma Detection using Custom CNN

## Problem Statement
The goal of this project is to build a custom Convolutional Neural Network (CNN) model that can accurately detect melanoma. Melanoma is a serious form of skin cancer responsible for 75% of skin cancer-related deaths. Early detection is crucial for effective treatment, and this project aims to build a model that can help dermatologists by identifying melanoma from images.

This project uses a dataset of 2357 images of malignant and benign oncological diseases, sourced from the International Skin Imaging Collaboration (ISIC). The dataset is classified into nine disease categories:
- Actinic keratosis
- Basal cell carcinoma
- Dermatofibroma
- Melanoma
- Nevus
- Pigmented benign keratosis
- Seborrheic keratosis
- Squamous cell carcinoma
- Vascular lesion

## Dataset
The dataset can be downloaded from [here](#). It consists of skin lesion images that are labeled based on the disease category. The class distribution is imbalanced, with melanomas and moles being slightly dominant in the dataset.

## Project Pipeline
1. **Data Reading & Understanding**
   - The dataset is loaded, and paths for training and test images are defined.

2. **Dataset Creation**
   - The training dataset is split into train and validation sets with a batch size of 32.
   - Images are resized to 180x180 pixels for uniformity.

3. **Dataset Visualization**
   - Sample images from all nine classes are visualized to understand the dataset distribution and characteristics.

4. **Model Building & Training**
   - A custom CNN model is built to detect the nine classes.
   - Images are rescaled to normalize pixel values between (0,1).
   - The model is trained for approximately 20 epochs using an appropriate optimizer and loss function.
   - Model performance is evaluated for overfitting or underfitting, and data augmentation strategies are applied if necessary.

5. **Data Augmentation**
   - The dataset is augmented to improve model generalization and reduce overfitting or underfitting issues.
   - The model is retrained on the augmented dataset for ~20 epochs.

6. **Class Distribution Analysis**
   - Class distribution is examined to identify under-represented classes.

7. **Handling Class Imbalance**
   - The `Augmentor` library is used to rectify class imbalances in the training dataset.
   - The model is retrained on the balanced dataset for ~30 epochs.

8. **Final Model Evaluation**
   - Model performance is assessed after handling class imbalance, and key findings are documented.

## Requirements
To run this project, you need the following dependencies:
- Python 3.x
- TensorFlow
- NumPy
- Matplotlib
- Augmentor
- Google Colab (recommended for GPU runtime)

Install the dependencies using:

```bash
pip install tensorflow numpy matplotlib Augmentor
```

## How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/melanoma-detection-cnn.git
   ```
2. Download the dataset from [here](#) and place the zip file in sample_data in content (in Colab).
3. Run the `ipynb` notebook in Google Colab.

## Results
- The model was trained for multiple epochs on the original and augmented datasets.
- **Overfitting/Underfitting and Class Imbalance**: The model demonstrates strong performance, achieving a maximum accuracy of around 85% for both training and validation data. Addressing class imbalance and applying data augmentation have effectively improved the model’s accuracy while mitigating overfitting. Overall, this CNN model is well-suited for predicting skin cancer and shows promising generalization to new data.
Training accuracy and validation accuracy increases. Model overfitting issue is solved. Class rebalance helps in augmentation and achieving the best Training and validation accuracy.


## Conclusion
This project successfully builds a CNN-based model for melanoma detection, contributing to the early detection of skin cancer. With further improvements, this model has the potential to assist dermatologists in identifying melanoma and other skin diseases with greater accuracy.
