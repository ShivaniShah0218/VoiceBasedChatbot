"""
    File Name: webapp.py
    Purpose: Frontend for the RAG based chatbot
    Author: Shivani Shah
    Created At: 08/08/2025

"""
import gradio as gr
from stt import stt_inference
from tts import tts_inference
from rag_chatbot import chatbot
from typing import List, Union
import logging

logging.basicConfig(level=logging.INFO)

sess_obj=chatbot.Chatbot()

def upload_pdfs(files):
    """
        Purpose: Build faiss index for the uploaded pdfs
        Input: Files to be used for reference
    """
    try:
        if not files:
            return "No files uploaded"
        sess_obj.build_faiss_index_from_pdfs(files)
        return f"{len(files)} PDFs processed and indexed"
    except Exception as e:
        logging.error("Error in building faiss index from pdfs "+e)
        return "Error in building faiss index from pdfs "+e 

def text_to_response(text_input: str):
    """
        Purpose: Get response for query asked in text
        Input: Text Input
    """
    try:
        logging.info("Getting the response for the query asked in text")
        response = sess_obj.get_response(text_input)
        audio_path = tts_inference.text_to_speech(response)
        return [(text_input, response)], audio_path
    except Exception as e:
        logging.error(f"Error during STT-TTS: {e}")
        return [("Error", "Something went wrong. Please try again.")], None


def stt_tts(audio_tp):
    try:
        logging.info("Getting the response for the query asked in audio")
        transcripted_text = stt_inference.transcribe(audio_tp)
        response = sess_obj.get_response(transcripted_text)
        audio_path = tts_inference.text_to_speech(response)
        return [(transcripted_text, response)], audio_path
    except Exception as e:
        logging.error(f"Error during STT-TTS: {e}")
        return [("Error", "Something went wrong. Please try again.")], None



with gr.Blocks(title="VoiceBot with PDF-based RAG") as demo:
    gr.Markdown("### 🎙️ VoiceBot powered by PDF Knowledge")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📄 Upload Knowledge Base")
            pdf_upload = gr.File(label="Upload PDFs", file_types=[".pdf"], file_count="multiple")
            upload_btn = gr.Button("Build Knowledge Base")
            upload_status = gr.Textbox(label="Upload Status")
            upload_btn.click(fn=upload_pdfs, inputs=[pdf_upload], outputs=[upload_status])

        with gr.Column(scale=2):
            gr.Markdown("### 💬 Chat Interface")
            cb = gr.Chatbot(label="Chat History")

            # Voice Input Section
            with gr.Tabs():
                with gr.TabItem("🎤 Voice"):
                    audio_input = gr.Audio(sources="microphone", type="numpy", label="Speak Your Question")
                    voice_submit = gr.Button("Submit Voice")
                    voice_submit.click(stt_tts, inputs=[audio_input], outputs=[cb, gr.Audio(type="filepath")])
            
                with gr.TabItem("⌨️ Text"):
                    text_ip = gr.Textbox(label="Type Your Question")
                    text_submit = gr.Button("Submit")
                    text_submit.click(text_to_response, inputs=[text_ip], outputs=[cb, gr.Audio(type="filepath")])


demo.launch()