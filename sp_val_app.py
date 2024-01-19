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
    "Nelson_Mandela"
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
    # Load the saved model
        # Load the recorded audio
    fs, audio = read('output.wav')

    # Extract MFCC features from the audio
    features = []
    mfccs = librosa.feature.mfcc(y=audio, sr=fs, n_mfcc=10)
    features.append(mfccs.T)
    features = np.array(features)
    
    num_of_channels = features.shape[-1]
    model = LSTMModel(num_of_channels, 128, len(speaker_names)).to(device)
    model.load_state_dict(torch.load('lstm_model.pth'))
    model.eval()



    # Use the model to predict the speaker
    with torch.no_grad():
        features_tensor = torch.tensor(features).float().to(device)
        preds = model(features_tensor)
        _, predicted = torch.max(preds.data, 1)

    # Get the claimed speaker name from the text field
    claimed_speaker_name = speaker_name_entry.get()

    # Load the trained LabelEncoder from a file
    with open('encoder.pkl', 'rb') as f:
        encoder = pickle.load(f)
    
    # Get the predicted speaker name
    predicted_speaker_name = encoder.inverse_transform(predicted.cpu().numpy())

    # Check if the predicted speaker name matches the claimed speaker name
    if predicted_speaker_name == claimed_speaker_name:
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