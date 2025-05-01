YourTTS Zero-Shot Voice Cloning with Sentiment Conditioning

This project implements a zero-shot voice cloning system using [Coqui TTS (YourTTS)](https://github.com/coqui-ai/TTS), enriched with sentiment conditioning and a rigorous trustworthiness evaluation framework.


## Features

- Clone voice from **any public YouTube video** using just a name or URL
- Generate speech in the **cloned voice** based on **user-provided text**
- **Condition** voice generation on detected **sentiment** of the input text
- Evaluate trustworthiness through:
  - **Robustness Testing** under multiple noise conditions
  - **Speech Intelligibility** via **Word Error Rate (WER)**
  - **Perceptual Quality** estimation via **MOS**
  - (Optional) **Deepfake Detection** evaluation


## Installation

Ensure you're using **Python 3.10** (not 3.12 or higher, which is incompatible with TTS).


# Create virtual environment (if desired)
python -m venv tts_env
.\tts_env\Scripts\activate  # Windows

# Install required packages
pip install TTS yt-dlp transformers pydub
pip install jiwer librosa soundfile torchaudio speechbrain
pip install youtube-search-python
# (Optional for future): pip install deepfake-audio-detection


## Usage

Run the script:

python voicesyn.py

Enter YouTube video URL for voice cloning. Example:

https://youtu.be/R-RaiQ7-zHA?si=eKUYPRnrc_OLET_T

Provide the text to be spoken by the cloned voice

The system will:

Extract and preprocess the speaker's voice

Analyze sentiment from your text

Synthesize speech using YourTTS

Apply noise types for robustness testing

Measure Word Error Rate (WER) and MOS scores

## Project Workflow

Download speaker voice sample – Extracts audio from a YouTube video.

Convert MP3 to WAV (16kHz) – Ensures compatibility with the TTS model.

Perform sentiment analysis – Determines the emotional tone of the text.

Map sentiment to emotion – Applies "Happy", "Sad", or "Neutral" conditioning.

Generate speech using YourTTS – Produces voice output in the cloned speaker’s voice.

Play the synthesized audio – Outputs the final speech.


## Datasets Used

1. YourTTS Pre-Trained Model Datasets (Voice Cloning)

YourTTS has been trained using various multi-speaker datasets, ensuring robust zero-shot voice cloning capabilities. The key datasets include:

VCTK (Voice Cloning Toolkit Dataset) – A collection of 109 speakers from various English accents, making it ideal for multi-accent voice synthesis.

LibriTTS – Derived from LibriSpeech, a large corpus of audiobooks, contributing to high-quality TTS synthesis.

Common Voice (Mozilla) – A diverse, open-source multilingual dataset, enhancing the model’s ability to clone voices from different accents and languages.

2. Sentiment Analysis Model Dataset

Your project leverages Hugging Face’s sentiment-analysis pipeline, which is pre-trained on large sentiment-labeled datasets:

Stanford Sentiment Treebank (SST-2) – Used for binary sentiment classification (Positive/Negative).

IMDB Reviews Dataset – Contains 50,000 movie reviews, useful for detecting nuanced sentiment.

Amazon Reviews Dataset – A large dataset of product reviews, covering a variety of sentiment polarities.

3. YouTube Audio Data (Dynamic Dataset)

Since the user provides a YouTube video URL, the extracted audio acts as a custom speaker dataset.

This dataset is dynamic, meaning it varies with each user input.

Extracted MP3 audio is converted to WAV and serves as the speaker reference for cloning.


## Sentiment-to-Emotion Mapping


## Requirements

TTS (YourTTS model)

yt-dlp (YouTube audio extraction)

transformers (Sentiment analysis)

pydub (Audio processing)


## Potential Enhancements

Improve noise reduction for better voice clarity.
 Expand sentiment analysis to detect nuanced emotions like anger or excitement.
 Build a web UI using Gradio or Streamlit for easy interaction.
 Fine-tune voice cloning with multiple speaker samples for accuracy.


## Contributers

Dhanush Kurapati U62623479

Anusha Pasupuleti U08296156

Bindu Lekha Raavi U45598958


## References

Fatima M. Inamdar, Ambesange, S., Mane, R., Hussain, H., Wagh, S., & Lakhe, P. (2023). Voice Cloning Using Artificial Intelligence and Machine Learning: A Review. Journal of Advanced Zoology, 44(S-7), 419-427.