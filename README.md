# 🧠 SignLang-Final --- Sign Language Recognition System

> _"Translating gestures into words, bridging silence with
technology."_

---

## 📘 Overview

**SignLang-Final** is a deep learning--based system that recognizes
**American Sign Language (ASL)** gestures from images or videos. It
integrates **Convolutional Neural Networks (CNN)** (using
**VGG16**) with a **Flask web application** to classify sign
gestures and display results through a simple user interface.

The project serves as a base for developing intelligent communication
aids for hearing and speech-impaired individuals.

---

## 🏗️ Project Structure

SignLang-Final/ ├── app.py # Flask web application backend ├──
templates/ # Frontend HTML templates │ ├── index.html │ └── result.html
├── static/ \# Static assets (CSS, JS, images, models) │ ├── css/ │ ├──
js/ │ └── images/ ├── asl_vgg16_local.ipynb \# Notebook for training
VGG16 model ├── notebook20796be3f4.ipynb \# Experimental notebook /
alternate model ├── project_report.md \# Detailed project report ├──
requirements.txt \# Python dependencies └── README.md \# Base project
readme

yaml Copy code

---

## ⚙️ Setup & Installation

### 🧩 Prerequisites - Python 3.8 or above  - pip (Python package
manager)  - (Optional) NVIDIA GPU with CUDA for faster
training/inference

### 📦 Installation Steps

1. **Clone the repository** ```bash git clone
https://github.com/Sayanandh/SignLang-Final.git cd SignLang-Final Create
and activate a virtual environment

bash Copy code python -m venv venv venv\Scripts\activate \# On Windows
source venv/bin/activate \# On macOS/Linux Install dependencies

bash Copy code pip install -r requirements.txt Run the Flask app

bash Copy code python app.py Access the web interface Visit
http://localhost:5000 in your browser.

🧭 Workflow Overview 1️⃣ Input Stage The user uploads an image or video
of a sign through the web UI.

2️⃣ Preprocessing The backend resizes and normalizes input using OpenCV.

Optional segmentation / hand detection isolates the region of interest.

3️⃣ Model Inference A VGG16-based CNN classifies the preprocessed input
into one of the ASL gesture classes.

4️⃣ Output Stage The system maps the model's prediction to its
corresponding alphabet or word (e.g., "A", "B", "Hello").

The result is displayed back on the web UI.

🧠 Deep Learning Model Details Parameter Description Model VGG16
(Convolutional Neural Network) Framework TensorFlow / Keras Dataset
American Sign Language (A--Z) Input Image Size 64×64 or 224×224 pixels
Loss Function Categorical Crossentropy Optimizer Adam Output Activation
Softmax Notebook Used asl_vgg16_local.ipynb

🏋️ Training Steps Load dataset and split into training/test sets.

Initialize pre-trained VGG16 (with frozen convolutional layers).

Add dense layers for gesture classification.

Train for multiple epochs (e.g., 50--100).

Save trained model as .h5 for deployment in app.py.

🌐 Web Application Architecture mermaid Copy code flowchart TD A\[User
Uploads Image/Video\] --\> B\[Flask Backend (app.py)\] B --\>
C\[Preprocessing with OpenCV\] C \--\> D\[VGG16 Model Prediction\] D
\--\> E\[Predicted Gesture Class\] E \--\> F\[Frontend UI (Result
Page)\] 🧩 Flask Routes (Example) Route Method Description / GET Renders
the homepage (upload form) /predict POST Receives image/video input and
returns predicted sign /about GET Optional informational page about the
project

🧰 Key Components File Purpose app.py Handles web routes, file uploads,
and prediction logic templates/index.html Home page where user uploads
signs templates/result.html Displays classification results static/
Holds CSS, JS, images, and model weights asl_vgg16_local.ipynb Model
training and testing notebook project_report.md Contains dataset info,
evaluation, and conclusion requirements.txt Lists all Python libraries
used

🧪 Example Workflow User visits the web page and uploads an image.

Flask saves the file temporarily on the server.

Model loads (model = load_model(\'model.h5\')) and predicts the sign.

Prediction is converted into a label using a mapping dictionary.

The result page shows the detected gesture and confidence score.

📊 Results Summary High accuracy achieved on ASL alphabet dataset using
fine-tuned VGG16.

Model effectively classifies most hand gestures (A--Z).

Provides real-time results through the web interface.

(See project_report.md for detailed metrics and confusion matrix.)

🧩 Configuration Options Feature Description How to Modify Model
Architecture Replace VGG16 with ResNet/MobileNet Modify notebook and
retrain Dataset Path Location of ASL images Change dataset path in
notebook Flask Port Default: 5000 Edit app.run(port=XXXX) New Gestures
Add new sign classes Add data, retrain, and update label map Performance
Speed up inference Enable GPU / use model quantization

🧱 System Flow Diagram mermaid Copy code sequenceDiagram participant
User participant Frontend participant Backend participant Model
User->>Frontend: Uploads sign image/video Frontend->\>Backend: Sends
input request Backend->\>Model: Preprocess & Predict
Model--\>>Backend: Returns predicted class Backend-->>Frontend:
Sends response Frontend-->>User: Displays recognized sign 🚀 Future
Enhancements 🧩 Add video-based recognition (RNN/LSTM integration)

📹 Enable real-time webcam inference

🌍 Support for multiple sign languages (ISL, BSL, ASL)

🖐️ Integrate MediaPipe Hand Landmark Detection

🗣️ Add sign-to-text or speech translation using NLP models

⚡ Optimize model for mobile or edge devices (TensorFlow Lite)

🤝 Contribution Guidelines Contributions are welcome! Follow these steps
to add new features or improvements:

Fork this repository

Create a new branch

bash Copy code git checkout -b feature-branch Make your changes and test
locally

Commit and push

bash Copy code git commit -m \"Added new feature\" git push origin
feature-branch Create a Pull Request for review

📚 References TensorFlow Documentation

Keras Applications -- VGG16

OpenCV Library

ASL Alphabet Dataset (Kaggle)

Flask Web Framework

🧾 License This project is open-source under the MIT License. You are
free to use, modify, and distribute this project with proper
attribution.

✍️ Author 👨‍💻 Sayanandh Aneesh

🎓 BTech CSE Student @ SCMS SSET

💡 AI & ML | Computer Vision | Flutter | Backend Developer

🌐 GitHub: @Sayanandh

Built with ❤️ using Python, Flask, TensorFlow, and OpenCV

---


