# **Trash Classification using Deep Learning**

This repository contains a deep learning project for classifying trash into different categories using TensorFlow and Keras. The dataset used for training and evaluation is the [TrashNet dataset](https://huggingface.co/datasets/garythung/trashnet). The project also integrates MLOps tools like GitHub Actions, Weights & Biases (W&B), and Hugging Face Hub for automation, model tracking, and deployment.

---

## **Project Overview**
The objective of this project is to build an image classification model to identify the type of trash from an image. The workflow includes:
1. **Data Preparation**: Loading, preprocessing, and augmenting the dataset.
2. **Exploratory Data Analysis (EDA)**: Visualizing sample images and analyzing dataset properties.
3. **Model Training**: Building and training a Convolutional Neural Network (CNN) using TensorFlow/Keras.
4. **Model Evaluation**: Evaluating the model's performance and identifying areas for improvement.
5. **MLOps Integration**:
   - Model tracking and versioning with W&B.
   - Automation using GitHub Actions.
   - Deployment to Hugging Face Hub.

---

## **Setup and Reproduction**

Follow these steps to reproduce the results:

### **1. Clone the Repository**
```bash
git clone https://github.com/your-username/trash-classification.git
cd trash-classification
```
Install all required Python packages using requirements.txt:
```python
pip install -r requirements.txt
```
Log in to Weights & Biases:
```python
wandb login
```
Log in to Hugging Face Hub:
```python
huggingface-cli login
```
Launch the Jupyter Notebook and follow the steps in AI_engineer_intern_test_adamata.ipynb:
```
jupyter notebook AI_engineer_intern_test_adamata.ipynb
```
MLOps Workflow
1. GitHub Actions
The CI/CD pipeline is set up using GitHub Actions. It includes:

- Code validation and dependency installation.
- Model training and evaluation.
- Deployment to Hugging Face Hub.

2. Model Tracking with W&B
Weights & Biases is used for:
- Tracking training metrics (loss, accuracy, etc.).
- Visualizing model performance.
3. Model Deployment
- The trained model is published to the Hugging Face Hub for easy access and inference.

# **Dataset**
The dataset used in this project is the is the [TrashNet dataset](https://huggingface.co/datasets/garythung/trashnet), containing images of trash classified into categories such as:

- Cardboard
- Glass
- Metal
- Paper
- Plastic
- Trash

# **Results**
The model achieves an accuracy of 20% on the validation dataset. Further improvements are possible by addressing dataset imbalance and optimizing hyperparameters.
