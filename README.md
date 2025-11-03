# Speech-Emotion-Recognition-using-MFCC-and-Wav2Vec2

# 🗣️ Speech Emotion Recognition using MFCC and Wav2Vec2

This project focuses on detecting emotions from speech audio using two approaches:  
1. A **baseline MFCC-based neural network**  
2. A **transformer-based Wav2Vec2 model** for advanced audio feature extraction  

Trained and evaluated on the **RAVDESS** (Ryerson Audio-Visual Database of Emotional Speech and Song) dataset.

## 🎯 Objective
To classify emotions such as **happy, sad, angry, calm, fearful, disgust, surprised,** and **neutral** directly from voice recordings.

## 📂 Dataset
- **Dataset:** RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)  
- **Samples:** 1,440 `.wav` files  
- **Emotions:** Neutral, Calm, Happy, Sad, Angry, Fearful, Disgust, Surprised  
- **Source:** [Zenodo - RAVDESS Dataset](https://zenodo.org/records/1188976)

## 🧠 Models

### 🔹 MFCC Baseline Model
- **Features:** 40 MFCCs extracted using Librosa  
- **Model:** 3-layer fully connected neural network (PyTorch)  
- **Epochs:** 1000  
- **Accuracy:** **70%**

### 🔹 Wav2Vec2 Transformer Model
- **Pretrained Model:** facebook/wav2vec2-base-960h  
- **Feature Dimension:** 768  
- **Training Status:** Running  
- **Expected Accuracy:** 75–85%

## ⚙️ Requirements
```bash
pip install torch torchvision torchaudio transformers librosa scikit-learn tqdm matplotlib
```
##Tools & Frameworks

- Python, PyTorch
- Librosa, Torchaudio
- Hugging Face Transformers
- Scikit-learn, Matplotlib
- Google Colab (GPU runtime)

  
## 📈 Results Summary

| Model | Feature Type | Epochs | Accuracy | Status | Remarks |
|--------|---------------|---------|-----------|---------|----------|
| **MFCC + MLP** | 40 MFCCs | 1000 | **70%** | ✅ Completed | Strong handcrafted baseline with good generalization |
| **Wav2Vec2 + Classifier** | 768-D Transformer Embeddings | Running | – | ⏳ In Progress | Expected 75–85% accuracy once training completes |

## 👨‍💻 Author

**Mirza Shaheen Iqubal**  
Master’s Student in Data Science  
Friedrich-Alexander-Universität Erlangen–Nürnberg (FAU)

---

## 💬 Acknowledgements

- [RAVDESS Dataset on Zenodo](https://zenodo.org/records/1188976)  
- [Facebook AI – Wav2Vec2.0](https://ai.facebook.com/)  
- [Hugging Face Transformers](https://huggingface.co/transformers/)

