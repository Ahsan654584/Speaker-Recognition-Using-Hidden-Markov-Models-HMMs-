🎙️ Speaker Recognition using Hidden Markov Models (HMM)
This project implements a speaker recognition system using Hidden Markov Models (HMMs) and MFCC (Mel Frequency Cepstral Coefficients) for feature extraction. It can classify whether a given voice belongs to a target speaker or not, based on statistical modeling and log-likelihood scoring.

📌 Features
🔉 Extracts MFCC features from .wav audio files

📈 Trains multiple HMMs to find the best performing model

🧠 Selects the best model based on log-likelihood score gap

🧪 Tests and classifies new voice samples

📊 Visualizes score distributions using histograms

💾 Saves trained models and features using Pickle

📁 Project Structure
graphql
Copy
Edit
├── Dataset/
│   ├── ahsan_voice/        # Target speaker voice samples
│   └── other_voice/        # Other speakers' voice samples
├── your_mfcc.pkl           # Extracted MFCCs of target speaker
├── others_mfcc.pkl         # Extracted MFCCs of other speakers
├── best_hmm_model.pkl      # Saved trained HMM model
├── test.wav                # Test sample for prediction
├── speech_recognition_hmm.py
└── README.md
🚀 How to Run
Clone the repository:

bash
Copy
Edit
git clone https://github.com/yourusername/hmm-speech-recognition.git
cd hmm-speech-recognition
Install required dependencies:

bash
Copy
Edit
pip install -r requirements.txt
Add your .wav files in the appropriate folders inside the Dataset directory:

Your own voice: Dataset/ahsan_voice/

Other voices: Dataset/other_voice/

Run the script:

bash
Copy
Edit
python speech_recognition_hmm.py
🧪 How It Works
MFCCs are extracted from all voice samples.

Multiple HMMs are trained with varying numbers of hidden states.

Each model's performance is scored on both target and other voice samples.

The model with the highest gap in log-likelihood scores is selected.

New voice samples can then be predicted using this trained model.

📊 Graphs and Visualization
You will see histograms showing:

The distribution of log-likelihood scores for your voice vs others.

Vertical lines indicating average scores for both groups.

These help you visually interpret how well the model differentiates between the speakers.

📌 Results Summary
✅ Best number of HMM states: Typically between 6 and 9

📈 Highest score gap: Achieved when the model could distinctly recognize the target speaker’s voice

🧠 Classification: The system accurately identified unseen voice samples using learned statistical patterns

📚 Conclusion
This project highlights the power of Hidden Markov Models when combined with MFCC features for voice-based applications. It demonstrates how even simple statistical models can be highly effective for speaker identification in controlled environments.

📂 GitHub Repository
You can explore the full project, including code, data, and documentation here:

👉 GitHub Repository Link
(Replace with your actual GitHub repo URL after uploading)

👨‍💻 Author
Muhammad Ahsan Kareem
Final Year Engineering Student
Passionate about Signal Processing, AI, and Embedded Systems
