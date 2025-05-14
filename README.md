# 🌿 Plant Leaf Disease Detection using ML, DL, QML, and QNN

This full-stack web application detects whether a plant leaf is healthy or diseased based on an uploaded image. It integrates **Machine Learning (Random Forest)**, **Deep Learning (CNN)**, **Quantum Machine Learning (VQC)**, and **Quantum Neural Networks (EstimatorQNN)**. The frontend is built with **Streamlit**, backend with **Python**, and **MySQL** is used for data storage. DevOps tools like **GitHub Actions**, **Docker**, and **Kubernetes** automate testing and deployment. The project follows the **Agile Scrum** methodology.

---

## Key Features

- User Login & Registration (MySQL)
- Upload leaf images via Streamlit interface
- Predicts disease using:
  - **ML** (Random Forest)
  - **DL** (CNN with Keras)
  - **QML** (VQC via Qiskit)
  - **QNN** (EstimatorQNN via PyTorch + Qiskit)
- Final prediction by **majority voting**
- Stores predictions in MySQL database
- Continuous Integration via GitHub Actions
- Dockerized app, deployed with Kubernetes

---

## Models Used

- **Random Forest**: Trained on 512-length color histogram features
- **CNN (Convolutional Neural Network)**: Trained on 128×128 leaf images using Keras
- **QML (Quantum Machine Learning)**: Uses `ZZFeatureMap` + `VQC` from Qiskit
- **QNN (Quantum Neural Network)**: Hybrid model using `EstimatorQNN` and PyTorch

---

## Dataset

- **Source**: [PlantVillage Dataset](https://www.kaggle.com/datasets/emmarex/plantdisease)
- **Subset Used**: 100 images per class
- **Classes**: 7 (e.g., Healthy, Late Blight, Leaf Mold, etc.)
- **Image Size**: Resized to 128x128 pixels

---

## Tech Stack

| Layer      | Technology |
|------------|------------|
| Frontend   | Streamlit  |
| Backend    | Python     |
| ML/DL/QML/QNN | scikit-learn, Keras, Qiskit, PyTorch |
| Database   | MySQL      |
| DevOps     | GitHub, GitHub Actions, Docker, Kubernetes |

---

## Testing

- Artifacts include:
  - Test Plan
  - Traceability Matrix
  - Test Cases

---

