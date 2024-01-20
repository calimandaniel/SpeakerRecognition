import tkinter as tk
from tkinter import messagebox
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write, read
import torch
import torchaudio
import librosa
from cnn_model import CNN
from lstm_model import LSTMModel
import pickle
import os
# Global variable to store the audio
audio = None

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# Label of the speakers in the dataset
speaker_names = [
    "Benjamin_Netanyau",
    "Jens_Stoltenberg",
    "Julia_Gillard",
    "Magaret_Tarcher",
    "Nelson_Mandela"#,
    #"unknown"
]

def start_recording():
    global audio
    duration = 10  # seconds
    fs = 16000  # Sample rate
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    #sd.wait()  # Wait until recording is finished
    #messagebox.showinfo("Information","Start Recording")

def stop_recording():
    global audio
    sd.stop()  # Stop the recording
    write('output.wav', 16000, audio)  # Save the audio to a file
    messagebox.showinfo("Information","Stop Recording")

def verify_speaker():
    # Load the recorded audio
    #audio, sample_rate = librosa.load("output.wav")
    sound_file = os.path.join('.//dataset//audio//Julia_Gillard', "1.wav")
    audio, sample_rate = librosa.load(sound_file)
    
    claimed_speaker_name = speaker_name_entry.get()
    claimed_speaker_index = speaker_names.index(claimed_speaker_name)
    
    # Extract MFCC features from the audio
    features = []
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=10)
    mfccs = mfccs[:, :32]
    features.append(mfccs.T)
    features = np.array(features)
    
    num_of_channels = features.shape[-1]
    model = LSTMModel(num_of_channels, 128, 1).to(device)  # Only one output unit
    model.load_state_dict(torch.load(f'lstm_model_{claimed_speaker_index}.pth'))  # Load the model for the claimed speaker
    model.eval()

    # Use the model to predict the speaker
    with torch.no_grad():
        features_tensor = torch.tensor(features).float().to(device)
        preds = model(features_tensor)
        preds = torch.sigmoid(preds)  # Apply sigmoid to get probabilities

    # Load the anomaly detection model
    with open('isolation_forests.pkl', 'rb') as file:
        isolation_forests = pickle.load(file)

    # Get the IsolationForest and the scaler for the claimed speaker
    isolation_forest, scaler = isolation_forests[claimed_speaker_index]

    # Reshape and standardize the features
    features_2d = features.reshape(features.shape[0], -1)
    features_scaled = scaler.transform(features_2d)

    # Use the IsolationForest to predict if the speaker is an anomaly
    anomaly_score = isolation_forest.decision_function(features_scaled)

    threshold_lstm = 0.5
    threshold_anomaly = -0.2

    # Check if the predicted probability is above the threshold for the LSTM model
    # and if the anomaly score is below the threshold for the IsolationForest
    if preds.item() >= threshold_lstm and anomaly_score > threshold_anomaly:
        messagebox.showinfo("Information", "The claim is verified. The speaker is indeed " + claimed_speaker_name)
    else:
        messagebox.showinfo("Information", "The claim is not verified. The speaker is not " + claimed_speaker_name)
window = tk.Tk()

start_button = tk.Button(window, text="Start Recording", command=start_recording)
start_button.pack()

stop_button = tk.Button(window, text="Stop Recording", command=stop_recording)
stop_button.pack()

speaker_name_label = tk.Label(window, text="Claimed Speaker Name")
speaker_name_label.pack()

speaker_name_entry = tk.Entry(window)
speaker_name_entry.pack()

verify_button = tk.Button(window, text="Verify Speaker", command=verify_speaker)
verify_button.pack()

window.mainloop()