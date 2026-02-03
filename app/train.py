import os
import torch
import torchaudio
import glob
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from app.config import settings

# --- CONFIG ---
DATA_DIR_REAL = "data/real"  # Put real MP3/WAV files here
DATA_DIR_FAKE = "data/fake"  # Put AI MP3/WAV files here
MODEL_SAVE_PATH = "app/models/classifier_head.pkl"
# --------------

def load_wav2vec():
    print("Loading Wav2Vec2 model for feature extraction...")
    processor = Wav2Vec2Processor.from_pretrained(settings.MODEL_WAV2VEC)
    model = Wav2Vec2Model.from_pretrained(settings.MODEL_WAV2VEC)
    model.eval()
    return processor, model

def extract_features(file_path, processor, model):
    try:
        waveform, sr = torchaudio.load(file_path)
        # Resample to 16k
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(sr, 16000)
            waveform = resampler(waveform)
        
        # Mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        input_values = processor(waveform.squeeze(), sampling_rate=16000, return_tensors="pt", padding=True).input_values
        
        with torch.no_grad():
            outputs = model(input_values)
            # Use mean of last hidden state as the "voice fingerprint"
            embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
            
        return embeddings
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def main():
    if not os.path.exists(DATA_DIR_REAL) or not os.path.exists(DATA_DIR_FAKE):
        print(f"❌ Data directories not found!")
        print(f"Please create '{DATA_DIR_REAL}' and '{DATA_DIR_FAKE}' and add audio files.")
        return

    processor, model = load_wav2vec()
    
    X = [] # Features
    y = [] # Labels (0 = Human, 1 = AI)
    
    # 1. Process Real
    real_files = glob.glob(os.path.join(DATA_DIR_REAL, "*.*"))
    print(f"Processing {len(real_files)} Real files...")
    for f in real_files:
        feat = extract_features(f, processor, model)
        if feat is not None:
            X.append(feat)
            y.append(0) # Label 0 for Human
            
    # 2. Process Fake
    fake_files = glob.glob(os.path.join(DATA_DIR_FAKE, "*.*"))
    print(f"Processing {len(fake_files)} Fake files...")
    for f in fake_files:
        feat = extract_features(f, processor, model)
        if feat is not None:
            X.append(feat)
            y.append(1) # Label 1 for AI

    X = np.array(X)
    y = np.array(y)
    
    if len(X) == 0:
        print("No audio data processed. Exiting.")
        return

    # 3. Train
    print("Training Classifier...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Logistic Regression is simple, fast, and explainable
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    
    # 4. Evaluate
    preds = clf.predict(X_test)
    print("\n--- Results ---")
    print(f"Accuracy: {accuracy_score(y_test, preds):.4f}")
    print(classification_report(y_test, preds, target_names=['Human', 'AI']))
    
    # 5. Save
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    joblib.dump(clf, MODEL_SAVE_PATH)
    print(f"\n✅ Classifier saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    main()
