import numpy as np              # For numerical operations
import librosa                 # For audio loading and MFCC extraction
import soundfile as sf         # Required by librosa to read/write audio
from hmmlearn.hmm import GaussianHMM  # For training and using HMM
import joblib                  # For saving and loading trained models
import os

def extract_mfcc(audio_path, n_mfcc=13):
    y, sr = librosa.load(audio_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return mfcc.T


def predict_voice_live(audio_path, model, threshold):
    mfcc = extract_mfcc(audio_path)
    score = model.score(mfcc)
    print(f"🎙️ Test Voice Log-Likelihood: {score:.2f}")
    if score > threshold:
        print("🔓 Access Granted: This is your voice.")
    else:
        print("❌ Access Denied: Voice does not match.")
best_model = joblib.load("best_hmm_model.pkl")
predict_voice_live("TEST_NOTCORRECT_1.wav", best_model, threshold=-9000)