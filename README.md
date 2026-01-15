🧠 DeepfakeBench: Deepfake Detection Benchmark
📌 Overview

DeepfakeBench is an AI-driven benchmarking framework designed to evaluate and compare deepfake detection models on manipulated facial media.
The project focuses on building a reproducible pipeline for dataset preprocessing, model training, evaluation, and inference, enabling systematic analysis of deepfake detection performance.

This repository serves as both a research-oriented benchmark and a practical implementation for real-world deepfake detection.

🎯 Objectives

1. Develop a standardized pipeline for deepfake detection benchmarking
2. Train and evaluate CNN-based deepfake classifiers on benchmark datasets
3. Compare model performance using robust evaluation metrics
4. Enable real-time inference through a user-friendly interface
5. Support reproducible research and extensibility

🗂️ Project Structure
DeepfakeBench/
│
├── dataset/
│   ├── real/
│   └── fake/
│
├── models/
│   ├── cnn_model.py
│   └── xception_model.py
│
├── notebooks/
│   ├── data_preprocessing.ipynb
│   └── model_training.ipynb
│
├── app.py                # Streamlit web application
├── train.py              # Model training script
├── evaluate.py           # Model evaluation script
├── requirements.txt
└── README.md

🧪 Dataset
Primary Dataset: FaceForensics++
Contains both real and manipulated (deepfake) facial images/videos
Preprocessing includes:
Frame extraction from videos
Face detection and cropping
Image normalization and resizing
⚠️ Dataset files are not included in this repository due to size and licensing constraints.   

🧠 Models Implemented
Convolutional Neural Networks (CNN)
Transfer Learning Models:
1. XceptionNet
2. ResNet (optional extension)

📊 Evaluation Metrics
Accuracy
Precision
Recall
F1-score
Confusion Matrix
These metrics provide a balanced evaluation for imbalanced deepfake datasets.

🚀 Features
Modular and scalable pipeline
Research-friendly benchmarking framework
Real-time inference using Streamlit
Clean separation of training, evaluation, and deployment
Easily extensible for new datasets and models

▶️ How to Run
1️⃣ Clone the Repository
git clone https://github.com/USERNAME/DeepfakeBench.git
cd DeepfakeBench

2️⃣ Install Dependencies
pip install -r requirements.txt

3️⃣ Train the Model
python train.py

4️⃣ Evaluate the Model
python evaluate.py

5️⃣ Run the Web App
streamlit run app.py

🔬 Research Contribution
Provides a reproducible benchmark for deepfake detection
Helps analyze model robustness against manipulated media
Can be extended to support multi-dataset benchmarking

🛠️ Future Enhancements
Support for additional datasets (DFDC, Celeb-DF)
Video-level deepfake classification
Attention-based and Transformer models
Explainable AI (Grad-CAM)
Model robustness testing against adversarial attacks

👩‍💻 Author
Akansha Srivastava
B.Tech Computer Science Engineering
Interested in AI, Cyber Security, and Applied Machine Learning

📜 License
This project is intended for academic and research purposes.
