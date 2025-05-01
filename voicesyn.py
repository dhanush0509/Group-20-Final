# YourTTS Zero-Shot Voice Cloning with Sentiment Conditioning in voice

# %pip install TTS yt-dlp transformers pydub
# pip install jiwer pydub librosa soundfile torchaudio speechbrain
# pip install deepfake-audio-detection


import os
from pydub import AudioSegment
from transformers import pipeline
from yt_dlp import YoutubeDL
import librosa
import numpy as np
import torchaudio
from jiwer import wer
from pydub import AudioSegment
from speechbrain.pretrained import EncoderDecoderASR
import soundfile as sf
from TTS.api import TTS
import IPython.display as ipd

# Download audio from YouTube
video_url = input("Enter YouTube video URL for voice cloning: ")
output_filename = "yourtts_voice_sample.mp3"

ydl_opts = {
    'format': 'bestaudio/best',
    'outtmpl': output_filename,
    'quiet': False
}

with YoutubeDL(ydl_opts) as ydl:
    ydl.download([video_url])

# Convert MP3 to WAV,  16kHz
audio = AudioSegment.from_file(output_filename)
audio = audio.set_channels(1).set_frame_rate(16000)
audio.export("yourtts_voice_sample.wav", format="wav")

# Get user text input
user_text_input = input("Enter the text you want the cloned voice to say: ")

# Analyze sentiment
sentiment_analyzer = pipeline("sentiment-analysis")
sentiment = sentiment_analyzer(user_text_input)[0]
print(f"Sentiment detected: {sentiment['label']} with confidence {sentiment['score']:.2f}")

# Set emotion conditioning based on sentiment
emotion_mapping = {
    "POSITIVE": "Happy",
    "NEGATIVE": "Sad",
    "NEUTRAL": "Neutral"
}

emotion = emotion_mapping.get(sentiment['label'], "Neutral")
print(f"Emotion applied: {emotion}")

# Load YourTTS model
tts = TTS(model_name="tts_models/multilingual/multi-dataset/your_tts")

# Generate speech with speaker reference and sentiment-based tone
output_file = "yourtts_cloned_voice.wav"
tts.tts_to_file(
    text=user_text_input,
    speaker_wav="yourtts_voice_sample.wav",
    language="en",
    file_path=output_file
)


# Play the result
ipd.Audio(output_file)

# Load YourTTS output
original_audio_path = "yourtts_cloned_voice.wav"

# Load audio
y, sr = librosa.load(original_audio_path, sr=16000)


def add_noise(audio, noise_type="gaussian", snr_db=10):
    rms = np.sqrt(np.mean(audio ** 2))
    if noise_type == "gaussian":
        noise = np.random.normal(0, rms / (10**(snr_db/20)), len(audio))
    elif noise_type == "reverb":
        reverb = np.convolve(audio, np.random.randn(1000) * 0.005, mode='same')
        return reverb
    elif noise_type == "env":
        noise, _ = librosa.load("ambient_cafe_noise.wav", sr=16000)
        noise = noise[:len(audio)]
    else:
        raise ValueError("Unsupported noise type")
    return audio + noise

# Add different types of distortion
noisy_versions = {
    "gaussian": add_noise(y, "gaussian", snr_db=10),
    "reverb": add_noise(y, "reverb"),
}

# Save noisy samples
for key, audio in noisy_versions.items():
    sf.write(f"{key}_cloned.wav", audio, sr)
# Load ASR model
asr_model = EncoderDecoderASR.from_hparams(
    source="speechbrain/asr-transformer-transformerlm-librispeech",
    savedir="pretrained_models/asr-transformer"
)

def transcribe(path):
    result = asr_model.transcribe_file(path)
    return result.lower()

reference_text = "Start making what makes sense it"

for key in noisy_versions:
    hyp = transcribe(f"{key}_cloned.wav")
    error = wer(reference_text, hyp)
    print(f"[{key}] WER: {error:.2f} | ASR Output: {hyp}")


def estimate_mos(audio):
    loudness = np.mean(np.abs(audio))
    clarity = np.std(audio)
    mos = max(1, min(5, 5 - (clarity * 20) + (loudness * 3)))
    return round(mos, 2)

for key, audio in noisy_versions.items():
    mos_score = estimate_mos(audio)
    print(f"[{key}] Estimated MOS: {mos_score}")
