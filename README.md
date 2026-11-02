<div align="center">

# 🐾 PawVision AI — Deep Learning Dog vs Cat Classifier

<p align="center">
  <img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=24&pause=1000&color=00F5FF&center=true&vCenter=true&width=650&lines=🐾+Next-Gen+CNN+Animal+Classifier;⚡+Powered+by+TensorFlow+%26+Streamlit;🎯+High-Precision+Binary+Classification;✨+Modern+Futuristic+Glassmorphism+UI" alt="Typing SVG" />
</p>

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-D00000?style=for-the-badge&logo=keras&logoColor=white)](https://keras.io)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-00F5FF?style=for-the-badge)](LICENSE)

<br/>

<p align="center">
  <b>PawVision AI</b> is an end-to-end Deep Learning application that distinguishes between dogs and cats in real-time with high confidence, wrapped in a cyberpunk glassmorphism web interface.
</p>

[✨ Live Features](#-key-features) • [🧠 Neural Architecture](#-neural-network-architecture) • [🚀 Quick Start](#-quick-start) • [📊 Results & Evaluation](#-model-evaluation--training-results) • [🔮 Workflow](#-end-to-end-workflow)

</div>

---

## 📸 Application Previews

<div align="center">

| 🏠 Clean Hero Dashboard | 🎯 Real-Time AI Prediction |
|:---:|:---:|
| <img src="assets/ui/homepage.png" width="450" alt="PawVision AI Dashboard"/> | <img src="assets/ui/prediction page.png" width="450" alt="AI Prediction Card"/> |

</div>

---

## ✨ Key Features

- **⚡ Fast Inference**: Instant predictions powered by an optimized Convolutional Neural Network (CNN).
- **🎨 Glassmorphism Dark UI**: Built with Streamlit, custom CSS animations, neon glows, and gradient typography.
- **📊 Real-time Confidence Gauges**: Dynamic prediction progress bars and exact probability scores for both classes.
- **🛡️ Robust Input Handling**: Automatic resizing to 256×256, RGB normalization, and support for JPG, JPEG, and PNG.
- **🧩 Production-Ready Caching**: Leverages `@st.cache_resource` for zero-redundancy model loading.

---

## 🔮 End-to-End Workflow

```mermaid
flowchart LR
    A["📷 User Image Upload"] --> B["🔄 Preprocessing (256x256, /255)"]
    B --> C["🧠 Deep CNN Pipeline"]
    C --> D["⚡ Sigmoid Output [0.0 - 1.0]"]
    D -->|"< 0.5"| E["🐱 CAT Detected"]
    D -->|">= 0.5"| F["🐶 DOG Detected"]
    E --> G["📊 Glassmorphism UI Result"]
    F --> G
```

<div align="center">
  <img src="assets/workflow/workflow.png" width="850" alt="PawVision AI Architecture Workflow"/>
</div>

---

## 🧠 Neural Network Architecture

The classifier is built using a custom multi-stage **Sequential CNN** trained on Kaggle's Dogs vs Cats dataset.

```
Input Image (256 x 256 x 3)
   │
   ├── [Conv2D: 32 filters (3x3)] ──> [Batch Normalization] ──> [MaxPool2D (2x2)]
   │
   ├── [Conv2D: 64 filters (3x3)] ──> [Batch Normalization] ──> [MaxPool2D (2x2)]
   │
   ├── [Conv2D: 128 filters (3x3)] ──> [Batch Normalization] ──> [MaxPool2D (2x2)]
   │
   ├── [Flatten Layer]
   │
   ├── [Dense: 128 units, ReLU] ──> [Dropout: 0.1]
   │
   ├── [Dense: 64 units, ReLU]  ──> [Dropout: 0.1]
   │
   └── [Dense: 1 unit, Sigmoid] ──> Output Probability
```

<details>
<summary><b>🔍 Click here to view layer specifications</b></summary>

| Layer Type | Filters / Units | Kernel / Pool | Activation | Regularization |
|:---|:---:|:---:|:---:|:---|
| **Conv2D + BatchNorm** | 32 | 3 × 3 | ReLU | Batch Normalization |
| **MaxPooling2D** | — | 2 × 2 (Stride 2) | — | — |
| **Conv2D + BatchNorm** | 64 | 3 × 3 | ReLU | Batch Normalization |
| **MaxPooling2D** | — | 2 × 2 (Stride 2) | — | — |
| **Conv2D + BatchNorm** | 128 | 3 × 3 | ReLU | Batch Normalization |
| **MaxPooling2D** | — | 2 × 2 (Stride 2) | — | — |
| **Flatten** | — | — | — | — |
| **Dense** | 128 | — | ReLU | Dropout (0.1) |
| **Dense** | 64 | — | ReLU | Dropout (0.1) |
| **Output (Dense)** | 1 | — | Sigmoid | Binary Crossentropy |

</details>

---

## 📊 Model Evaluation & Training Results

<div align="center">
  <img src="assets/graphs.png" width="750" alt="Loss & Accuracy Graphs"/>
</div>

<div align="center">

| Metric | Training | Validation |
|:---|:---:|:---:|
| **Loss Function** | Binary Cross-Entropy | Binary Cross-Entropy |
| **Optimizer** | Adam (`lr=0.001`) | Adam |
| **Batch Size** | 32 | 32 |
| **Target Size** | 256 × 256 | 256 × 256 |

</div>

---

## 📁 Repository Structure

```bash
📦 Cat-and-Dog-Classifier
├── 📂 assets
│   ├── 📂 ui                     # Dashboard & prediction screenshots
│   │   ├── homepage.png
│   │   └── prediction page.png
│   ├── 📂 workflow               # Pipeline workflow diagrams
│   │   └── workflow.png
│   ├── graphs.png               # Training accuracy & loss curves
│   ├── cat.jpg                  # Sample test images
│   └── dog.jpg
├── 📂 models
│   └── dog_cat_cnn.keras        # Trained Keras model weights (download/place here)
├── 📂 notebook
│   └── cat_and_dog_classifier.ipynb  # End-to-end training notebook
├── 📄 .gitignore                # Production ignore rules
├── 📄 app.py                    # Streamlit web application
└── 📄 README.md                 # Interactive documentation
```

---

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/cat-and-dog-classifier.git
cd cat-and-dog-classifier
```

### 2. Create and Activate Virtual Environment
```bash
# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install streamlit tensorflow numpy pillow opencv-python matplotlib
```

### 4. Ensure Model File Exists
Place your trained `dog_cat_cnn.keras` model inside the `models/` directory:
```
models/dog_cat_cnn.keras
```

### 5. Launch the Streamlit App
```bash
streamlit run app.py
```

Navigate to `http://localhost:8501` in your browser to interact with PawVision AI.

---

## ❓ Frequently Asked Questions

<details>
<summary><b>🐱 Why does output < 0.5 indicate Cat?</b></summary>
<p>
The dataset directories are sorted alphabetically by default when imported using <code>tf.keras.utils.image_dataset_from_directory</code>:
<ul>
  <li><b>Class 0</b> = <code>cat</code></li>
  <li><b>Class 1</b> = <code>dog</code></li>
</ul>
Hence, predictions closer to 0 represent cats, while values closer to 1 represent dogs.
</p>
</details>

<details>
<summary><b>⚡ Can I run this with GPU acceleration?</b></summary>
<p>
Yes! On Linux/macOS or WSL2 on Windows, TensorFlow automatically utilizes CUDA-compatible GPUs. On native Windows, TensorFlow versions 2.11+ run efficiently on multi-core CPUs via oneDNN.
</p>
</details>

<details>
<summary><b>📦 Why are <code>.keras</code> and <code>.zip</code> files in .gitignore?</b></summary>
<p>
GitHub restricts single file uploads larger than 100MB. The trained model is ~178MB and the dataset archive is ~1.14GB. Use Git LFS or external cloud storage releases to distribute heavy weights.
</p>
</details>

---

## 📄 License

Distributed under the **MIT License**. See `LICENSE` for more information.

<div align="center">
  <sub>Built with ❤️ using TensorFlow, Keras, and Streamlit. Star ⭐ this repository if you found it useful!</sub>
</div>

