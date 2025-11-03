# Speech-Emotion-Recognition-using-MFCC-and-Wav2Vec2

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

## 🧠 Approaches Implemented

### 🔹 1. MFCC-based Baseline Model (Still Improving)
This model extracts **Mel-Frequency Cepstral Coefficients (MFCCs)** and feeds them into a simple feedforward neural network.

**Steps:**
1. Load RAVDESS audio files using `librosa`
2. Extract 40 MFCC features
3. Normalize using `StandardScaler`
4. Train a 3-layer **Fully Connected Neural Network (PyTorch)**  
   - Linear(40 → 128) → ReLU → Dropout  
   - Linear(128 → 64) → ReLU → Dropout  
   - Linear(64 → 8) → Softmax  
5. Evaluate on 25% test split

**Results:**
- **Epochs:** 1000  
- **Accuracy:** **~70%**  
- **Observation:** Stable learning and good generalization for small dataset.

---

### 🔹 2. Transformer-based Wav2Vec2 Model (ongoing)
This advanced model uses **pretrained self-supervised embeddings** from `facebook/wav2vec2-base-960h`.

**Steps:**
1. Load `.wav` data and resample to 16kHz  
2. Extract 768-dimensional audio embeddings via Wav2Vec2  
3. Train a small classification head on top of frozen embeddings  
4. Compare results with MFCC model  

**Expected Outcome:**
- Transformer models outperform classical MFCC features  
- **Predicted Accuracy:** ~80–85% after training completion  

---


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

