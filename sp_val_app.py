import tkinter as tk
from tkinter import messagebox
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write, read
import torch
from tkinter import ttk
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
# !!! Adjust the list of speaker names to match the speakers in your dataset
speaker_names = [
    "Benjamin_Netanyau",
    "Jens_Stoltenberg",
    "Julia_Gillard",
    "Magaret_Tarcher",
    "Nelson_Mandela",
    "Vilma"
]

def start_recording():
    """
    This function starts the audio recording for a specified duration and sample rate.
    """
    global audio
    duration = 5  # seconds
    fs = 16000  # Sample rate
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)

def stop_recording():
    """
    This function stops the audio recording, saves the recorded audio to a file, and displays a message box.
    """
    global audio
    sd.stop()  # Stop the recording
    write('output.wav', 16000, audio)  # Save the audio to a file
    messagebox.showinfo("Information","Stop Recording")

def verify_speaker():
    """
    This function verifies the identity of a speaker using a pre-trained LSTM model.

    The function performs the following steps:
    1. Loads an audio file named "output.wav".
    2. Retrieves the claimed speaker's name from a GUI entry field and finds the corresponding index in the list of known speaker names.
    3. Extracts MFCC features from the audio file.
    4. Creates an instance of a LSTM model, loads the model parameters from a file named 'lstm_model.pth', and sets the model to evaluation mode.
    5. Uses the model to predict the speaker and applies softmax to the outputs to get probabilities.
    6. Loads the trained LabelEncoder from a file to get the predicted speaker name.
    7. Checks if the predicted speaker name matches the claimed speaker name and the maximum confidence is below a certain threshold.
    8. Displays a message box with the verification result.
    """
    audio, sample_rate = librosa.load("output.wav")

    claimed_speaker_name = speaker_name_entry.get()
    claimed_speaker_index = speaker_names.index(claimed_speaker_name)
    
    # Extract MFCC features from the audio
    features = []
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=10)
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

        # Apply softmax to the outputs to get probabilities
        probs = torch.nn.functional.softmax(preds, dim=1)
        claimed_speaker_confidence = probs[0][claimed_speaker_index].item()
    
    # Load the trained LabelEncoder from a file
    with open('encoder_lstm.pkl', 'rb') as f:
        encoder = pickle.load(f)
    
    # Get the predicted speaker name
    predicted_speaker_name = encoder.inverse_transform(predicted.cpu().numpy())
    
    threshold = 0.5

    # Check if the predicted speaker name matches the claimed speaker name and the maximum confidence is below the threshold
    if claimed_speaker_confidence >= threshold and predicted_speaker_name == claimed_speaker_name:
        messagebox.showinfo("Information", "The claim is verified. The speaker is indeed " + claimed_speaker_name)
    else:
        messagebox.showinfo("Information", "The claim is not verified. The speaker is not " + claimed_speaker_name)

# Create the main window and set its properties
window = tk.Tk()
window.title("Speaker Verification App")
window.geometry("400x300")
window.configure(bg='light blue')

# Create a style object
style = ttk.Style()

# Configure the style of the buttons
style.configure('TButton', font=('calibri', 10, 'bold'), borderwidth='4')

# Configure the style of the labels
style.configure('TLabel', font=('calibri', 12, 'bold'), background='light blue')

start_button = ttk.Button(window, text="Start Recording", command=start_recording)
start_button.pack(pady=10)

stop_button = ttk.Button(window, text="Stop Recording", command=stop_recording)
stop_button.pack(pady=10)

speaker_name_label = ttk.Label(window, text="Claimed Speaker Name")
speaker_name_label.pack(pady=10)

speaker_name_entry = ttk.Entry(window)
speaker_name_entry.pack(pady=10)

verify_button = ttk.Button(window, text="Verify Speaker", command=verify_speaker)
verify_button.pack(pady=10)

window.mainloop()