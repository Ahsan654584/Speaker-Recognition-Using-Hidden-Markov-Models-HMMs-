# -----------------------------------------------
# 📦 Imports
# -----------------------------------------------
import os
import numpy as np
import librosa
import pickle
import matplotlib.pyplot as plt
import librosa.display
from hmmlearn import hmm

# -----------------------------------------------
# 🎙️ Step 1: Extract MFCCs from Folder
# -----------------------------------------------

def extract_mfcc_from_folder(folder_path):
    mfcc_list = []
    for file in os.listdir(folder_path):
        if file.endswith(".wav"):
            file_path = os.path.join(folder_path, file)
            y, sr = librosa.load(file_path, sr=16000)
            y = librosa.util.normalize(y)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).T
            mfcc_list.append(mfcc)
    return mfcc_list

your_mfccs = extract_mfcc_from_folder("Dataset/sibgha_voice")
others_mfccs = extract_mfcc_from_folder("Dataset/other_voice")

# Save features (optional)
with open("your_mfcc.pkl", "wb") as f: pickle.dump(your_mfccs, f)
with open("others_mfcc.pkl", "wb") as f: pickle.dump(others_mfccs, f)

# 🔍 Plot 1: MFCC heatmap for a sample file
sample_file = os.path.join("Dataset/sibgha_voice", os.listdir("Dataset/sibgha_voice")[0])
y, sr = librosa.load(sample_file, sr=16000)
mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

plt.figure(figsize=(10, 4))
librosa.display.specshow(mfcc, x_axis='time', sr=sr)
plt.colorbar()
plt.title("MFCCs of a Sample Voice")
plt.tight_layout()
plt.show()

# -----------------------------------------------
# 🤖 Step 2: Train and Select Best HMM
# -----------------------------------------------

def train_hmm(mfcc_data, n_components=5):
    X = np.concatenate(mfcc_data)
    lengths = [len(seq) for seq in mfcc_data]
    model = hmm.GaussianHMM(n_components=n_components, covariance_type='diag', n_iter=1000, tol=1e-4)
    model.fit(X, lengths)
    return model

def get_avg_score(model, mfcc_data):
    return np.mean([model.score(seq) for seq in mfcc_data])

best_model = None
best_gap = float('-inf')
best_scores = {}
gap_values = []
your_values = []
others_values = []
component_range = range(4, 11)

for n in component_range:
    model = train_hmm(your_mfccs, n_components=n)
    your_score = get_avg_score(model, your_mfccs)
    others_score = get_avg_score(model, others_mfccs)
    gap = your_score - others_score

    gap_values.append(gap)
    your_values.append(your_score)
    others_values.append(others_score)

    if gap > best_gap:
        best_gap = gap
        best_model = model
        best_scores = {"your_score": your_score, "others_score": others_score}

# 📊 Plot 2: Log-likelihood score vs # of states
plt.plot(component_range, gap_values, marker='o', label='Score Gap (Yours - Others)')
plt.plot(component_range, your_values, marker='x', linestyle='--', label='Your Score')
plt.plot(component_range, others_values, marker='x', linestyle='--', label='Others Score')
plt.xlabel('Number of HMM States')
plt.ylabel('Average Log-Likelihood')
plt.title('Model Selection: Log-Likelihood Scores')
plt.legend()
plt.grid(True)
plt.show()

# Save best model
with open("best_hmm_model.pkl", "wb") as f:
    pickle.dump(best_model, f)

# -----------------------------------------------
# 📈 Step 3: Score Distributions
# -----------------------------------------------

your_scores = [best_model.score(seq) for seq in your_mfccs]
others_scores = [best_model.score(seq) for seq in others_mfccs]

# 📊 Plot 3: Histogram of scores
plt.hist(your_scores, bins=10, alpha=0.6, label='Your Voice')
plt.hist(others_scores, bins=10, alpha=0.6, label='Others')
plt.axvline(np.mean(your_scores), color='blue', linestyle='dashed', label='Your Avg')
plt.axvline(np.mean(others_scores), color='red', linestyle='dashed', label='Others Avg')
plt.title("Voice Likelihood Score Distribution")
plt.xlabel("Log-Likelihood Score")
plt.ylabel("Frequency")
plt.legend()
plt.tight_layout()
plt.show()

# 📊 Plot 4: Threshold visualization
threshold = (best_scores['your_score'] + best_scores['others_score']) / 2
plt.hist(your_scores, bins=10, alpha=0.6, label='Your Voice')
plt.hist(others_scores, bins=10, alpha=0.6, label='Others')
plt.axvline(threshold, color='green', linestyle='--', label='Threshold')
plt.title("Decision Threshold")
plt.xlabel("Log-Likelihood Score")
plt.ylabel("Frequency")
plt.legend()
plt.tight_layout()
plt.show()

# -----------------------------------------------
# 🔍 Step 4: Prediction Function
# -----------------------------------------------

def predict_voice(file_path, model, threshold):
    y, sr = librosa.load(file_path, sr=16000)
    y = librosa.util.normalize(y)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).T
    score = model.score(mfcc)

    print(f"\n📣 Test voice score: {score:.2f}")
    if score >= threshold:
        print("✅ This is likely YOUR voice.")
    else:
        print("❌ This is NOT your voice.")

# 🔄 Example: Predict on a test file
predict_voice("test.wav", best_model, threshold)
