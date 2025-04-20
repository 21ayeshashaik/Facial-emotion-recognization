
# DeepFER: Facial Emotion Recognition using Deep Learning 😄😢😠

> A deep learning-based system to classify facial expressions into basic human emotions using Convolutional Neural Networks (CNNs).

---

## 📌 Overview

DeepFER (Deep Facial Emotion Recognition) is a machine learning project that classifies human emotions from facial images. This application can be used in areas like mental health analysis, smart surveillance, human-computer interaction, and more.

- 📷 **Input:** Grayscale facial images (e.g., 48x48 resolution)
- 🧠 **Output:** Predicted emotion (Happy, Sad, Angry, Surprise, Fear, Disgust, Neutral)

---

## 🔍 Features

- ✅ Real-time emotion detection using webcam (OpenCV)
- ✅ Trained CNN model with high accuracy on FER-2013 dataset
- ✅ Live emotion classification with UI overlay
- ✅ Visualized training history (accuracy/loss)

---

## 🧠 Emotions Detected

| Emotion     | Emoji     |
|-------------|-----------|
| Angry       | 😠        |
| Disgust     | 🤢        |
| Fear        | 😨        |
| Happy       | 😄        |
| Sad         | 😢        |
| Surprise    | 😲        |
| Neutral     | 😐        |

---

## 🗂️ Dataset

The model was trained on the **FER-2013** dataset.  
- Format: Grayscale, 48x48 px  
- Total Images: ~35000
- Classes: 7 emotion categories

Sample image:

![sample](images/sample_emotion.png)

---

## 🧱 Model Architecture

Built using **TensorFlow/Keras**, the CNN contains:

- Convolution layers with ReLU activation  
- MaxPooling layers  
- Dropout layers for regularization  
- Dense layers with Softmax output

```python
model = Sequential([
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(7, activation='softmax')
])
```

---

## 🚀 How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/21ayeshashaik/deepFER.git
cd deepFER
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the Model

```bash
python train.py
```

### 4. Run Real-Time Emotion Detection

```bash
python live_demo.py
```

> ✅ Make sure your **webcam is enabled** and **OpenCV** is properly installed.

---

## 🧪 Sample Output

![demo](images/demo.gif)

---

## 📈 Training Performance

- Optimizer: Adam  
- Loss Function: Categorical Crossentropy  
- Accuracy Achieved: **~%**

Training Curve:

![accuracy](images/training_accuracy.png)

---

## 🛠️ Future Work

- 🔁 Use transfer learning with VGGFace or MobileNet  
- ⚖️ Improve class balance with augmentation or weighted loss  
- 🧠 Explore multi-modal emotion recognition (e.g., audio + video)

---

## 🙌 Acknowledgements

- [FER-2013 Dataset](https://www.kaggle.com/datasets/msambare/fer2013)  
- TensorFlow & Keras libraries  
- OpenCV for real-time video processing

---

## 📬 Contact

**Shaik Ayesha**  
📧 shaikayesha2107@gmail.com  
🐱 GitHub: [@21ayeshashaik](https://github.com/21ayeshashaik)

---

## ⭐️ If you found this useful...

Leave a ⭐️ and share the repo to support the project!
