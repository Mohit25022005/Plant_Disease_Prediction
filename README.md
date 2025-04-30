# 🌿 Plant Disease Prediction using Deep Learning

This project aims to build a deep learning-based image classification model to detect and classify plant diseases from leaf images. Using Convolutional Neural Networks (CNNs), the model identifies various plant diseases to assist farmers and agriculturists in early diagnosis and prevention.

## 🚀 Live Demo

Try it here: [🌐 plantdisease-prediction.streamlit.app](https://plantdisease-prediction.streamlit.app/)

A web interface built using Streamlit allows users to upload leaf images and receive real-time disease predictions.

![Demo Screenshot](https://github.com/user-attachments/assets/fd923e0f-7c16-4dbf-99ef-1017335286ad)

## 🧠 Model Highlights

- Built with TensorFlow and Keras.
- Uses transfer learning (e.g., MobileNetV2 or EfficientNet) for faster training and better accuracy.
- Trained on the [New Plant Diseases Dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset).
- Achieves high precision and recall scores on test data.

## 📁 Dataset

- **Source**: [Kaggle - New Plant Diseases Dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)
- **Classes**: Healthy and multiple diseased conditions for crops like tomato, potato, maize, etc.
- **Preprocessing**: 
  - Image resizing and normalization
  - Augmentation for robust model training

## 📦 Tech Stack

- **Deep Learning**: TensorFlow, Keras
- **Data Processing**: NumPy, Pandas, OpenCV
- **Visualization**: Matplotlib, Seaborn
- **Web App**: Streamlit
- **Deployment**: [Streamlit Cloud](https://plantdisease-prediction.streamlit.app/)

## 🧪 Evaluation

- **Metrics Used**: Accuracy, Precision, Recall, F1-Score, Confusion Matrix
- **Results**: (Example)
