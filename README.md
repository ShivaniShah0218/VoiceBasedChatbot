# VoiceBasedChatbot
RAG-Driven Voice Interaction Assistant

This project demonstrates an end-to-end pipeline combining **ASR (Automatic Speech Recognition)**, **Language Modeling**, and **Text-to-Speech (TTS)** using lightweight and efficient models. It takes query in either form: audio or text, finds the relevant text from the files uploaded and generates the response and synthesizes the response into spoken response.

## 🔧 Components Used

### 🧠 1. Language Model: [`HuggingFaceTB/SmolLM-135M`](https://huggingface.co/HuggingFaceTB/SmolLM-135M)
- A small, efficient transformer-based language model (135M parameters).
- Used to generate or transform text based on transcribed speech input.
- Hugging Face Transformers-compatible.

### 🗣️ 2. ASR: [`faster-whisper`](https://github.com/guillaumekln/faster-whisper)
- Fast and memory-efficient speech-to-text model.
- Uses the `small` variant for balance between speed and accuracy.
- Converts audio input into text.

### 🔍 3. Vector Search: [`FAISS`](https://github.com/facebookresearch/faiss)
- Used to index and search a vector store of text/document embeddings.
- Enables semantic retrieval for contextual response generation.
- Ideal for knowledge base or memory-augmented dialogue systems.

### 🔊 4. TTS: [`coqui-tts`](https://github.com/coqui-ai/TTS)
- Used to synthesize speech from the processed/generated text.
- Supports multiple pre-trained voices and fine-tuning options.

## 📦 Installation

```bash
git clone https://github.com/ShivaniShah0218/VoiceBasedChatbot.git
cd VoiceBasedChatbot

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt
