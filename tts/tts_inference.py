"""
    File Name: tts_inference.py
    Purpose: Convert the response obtained from LLM to voice based output
    Author: Shivani Shah
    Created At: 08/08/2025

"""
from TTS.api import TTS
import tempfile
import logging

logging.basicConfig(level=logging.INFO)

logging.info("Load model for TTS")
tts=TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC").to("cpu")

def text_to_speech(text):
    """
        Purpose: Carry out text to speech
        Input: Text
        Output: Audio file path
    """
    try:
        logging.info("Converting Text to Speech")
        # Create a temporary WAV file in current directory
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False, dir="./generated_audio_files") as tmp_file:
            output_path = tmp_file.name
        logging.info("Audio saved in file '%s'"%(output_path))

        # Generate and save speech
        tts.tts_to_file(text=text, file_path=output_path)

        return output_path
    except Exception as e:
        logging.error("Error in converting Text-to-Speech "+e)
        return None
