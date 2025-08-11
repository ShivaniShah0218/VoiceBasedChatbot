"""
    File Name: stt_inference.py
    Purpose: Load transcription model and carry out transcription
    Author: Shivani Shah
    Created At: 08/08/2025

"""


from faster_whisper import WhisperModel
import tempfile
from scipy.io import wavfile
import logging

logging.basicConfig(level=logging.INFO)

logging.info("Loading Model for transcription")
model=WhisperModel("small",compute_type="int8",device="cpu")

def transcribe(audio_tp):
    """
        Purpose: Carry out transcription
        Input: Audio Bytes
        Output: Transcripted text
    """
    try:
        logging.info("Carrying out transcription for fetched audio bytes")
        sample_rate,audio=audio_tp
        with tempfile.NamedTemporaryFile(delete=False,dir="./audio_files/",suffix=".wav") as tmp:
            tmp_path=tmp.name
        wavfile.write(tmp_path, sample_rate, audio)
        segments,info=model.transcribe(tmp_path,beam_size=5)
        logging.info("Detected Language '%s' with probability %f"%(info.language,info.language_probability))
        return " ".join([segment.text for segment in segments])
    except Exception as e:
        logging.error("Error in transcription "+e)
        return "Error in transcription "+e
