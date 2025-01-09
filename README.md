#### ml-zoomcamp-capstone
# Image Classification
## Deep Learning Deployment with Streamlit & TensorFlow

---

## Project Overview
This capstone project is an **image classification** model that identifies six categories:  
**[ Buildings, Forest, Glacier, Mountain, Sea, and Street ]**.  

---

## Problem Description
This project aims to classify images into one of six scene categories using a **Deep Learning** model. The goal is to:
- Train a robust **image classifier** using **transfer learning**.
- Deploy the model as a **web application** using **Streamlit**.
- Package the deployment using **Docker** for easy reproducibility.

---

## Dataset
The dataset used for this project is the **Intel Image Classification Dataset**, which consists of labeled images for six different scene categories. The dataset is preprocessed and split into **training, validation, and test sets**.

Kaggel Link: 
[Intel Image Classification on Kaggle](https://www.kaggle.com/datasets/puneet6060/intel-image-classification)

---

## Exploratory Data Analysis (EDA)
To better understand the dataset, we performed an extensive EDA, which includes:
- **Data Structure**: Verified the organization of training, validation, and test sets.
- **Image/Label Shape**: Checked image consistency and label correctness.
- **Class Distribution**: Ensured balanced distribution across categories.
- **Visualize Pictures**: Displayed sample images to confirm dataset integrity.

For dataset analysis, refer to the **Jupyter Notebook** in the repository.

---

## Model Training & Implementation
- **Architecture**: Pretrained **Xception model** with fine-tuning.
- **Loss Function**: Categorical Cross-Entropy.
- **Optimizer**: Adam Optimizer.
- **Evaluation Metrics**: Accuracy, Validation Loss.
- **Data Augmentation**: Applied techniques like flipping, rotation, and brightness adjustment.

### Model Callbacks
- **ModelCheckpoint**: Saves the best-performing model.
- **EarlyStopping**: Stops training if no improvement is observed.
- **ReduceLROnPlateau**: Reduces learning rate on validation loss plateau.

---

## Deployment
This model is deployed using **Streamlit** and **Docker**.

### How It Works
1. **Upload an image** (JPG, PNG, JPEG).
2. **Select image processing method**:
   - Resize with padding (Maintain Aspect Ratio).
   - Central crop (Remove edges).
3. **Prediction**: The model classifies the image and outputs:
   - **Predicted Class**.
   - **Confidence Score**.

### Hosting & Infrastructure
- **Streamlit Cloud** for web deployment. **[Live Demo: [Image Classifier](https://ml-zoomcamp-capstone-hbfs8xbcqng2dg7w7q6yxe.streamlit.app/)]**
- **Google Drive (gdown)** for downloading model weights.
- **Docker** for containerized application deployment.

---

## Installation & Setup

### **1. Clone the Repository**
```bash
git clone https://github.com/Pei-Tong/ml-zoomcamp-capstone.git
cd ml-zoomcamp-capstone
```

### **2. Install Dependencies**
```bash
pip install -r requirements.txt
```

### **3. Train the model from scratch**
```bash
python train.py
```

### **4-1. Run Streamlit App**
```bash
streamlit run app.py
```

### **4-2. Run with Docker**
```bash
docker build -t image-classifier .
docker run -p 5000:5000 image-classifier
```

### **4-3. Run on Streamlit Cloud**
**Live Demo: [Image Classifier](https://ml-zoomcamp-capstone-hbfs8xbcqng2dg7w7q6yxe.streamlit.app/)**

---

##  Project Files

| File             | Description                                          |
|-----------------|------------------------------------------------------|
| `app.py`        | Streamlit UI for the image classification app.      |
| `train.py`      | Model training script using TensorFlow.             |
| `predict.py`    | Model inference script.                             |
| `service.py`    | BentoML service file for model serving.             |
| `Dockerfile`    | Docker setup for deployment.                        |
| `requirements.txt` | Required dependencies for the project.          |

---

##  Model Performance
The final model achieves high accuracy on the test dataset. The evaluation metrics are as follows:

| Metric               | Score |
|----------------------|-------|
| Training Accuracy   | 88.05%   |
| Validation Accuracy | 89.68%   |
| Test Accuracy       | 89.27%   |

---

## Web Interface
The following is a screenshot of the Customer Purchase Prediction app interface:

![image](https://github.com/user-attachments/assets/c0ce9a43-93bd-4246-b760-8eac46b432c5)

---

## Evaluation Criteria Mapping

| Criteria                         | Status                                      |
|----------------------------------|---------------------------------------------|
| **Problem Description**         | ✅ Described in README                     |
| **EDA**                         | ✅ Included dataset analysis in Jupyter Notebook |
| **Model Training**               | ✅ Used transfer learning, tuning, and augmentation |
| **Exporting Notebook to Script** | ✅ `train.py` provided for training         |
| **Reproducibility**              | ✅ Dataset link provided, setup instructions included |
| **Model Deployment**             | ✅ Streamlit & Docker used for deployment  |
| **Dependency Management**        | ✅ `requirements.txt` included             |
| **Containerization**             | ✅ Dockerfile provided                      |
| **Cloud Deployment**             | ✅ Hosted on Streamlit Cloud               |

---

##  Future Improvements
- Fine-tuning the model with larger datasets.
- Implementing real-time image classification.
- Deploying on cloud platforms like AWS/GCP.


