"""
    File Name: chatbot.py
    Purpose: RAG based chatbot
    Author: Shivani Shah
    Created At: 08/08/2025

"""
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import faiss
import numpy as np
import fitz
import logging

logging.basicConfig(level=logging.INFO)


class Chatbot:
    """
        Purpose: RAG based Chatbot
    """
    def __init__(self):
        """
            Purpose: Initialize embeddings and LLM
        """
        self.embedder=SentenceTransformer("all-MiniLM-L6-v2")
        self.llm=pipeline("text-generation",model="HuggingFaceTB/SmolLM-135M")#"microsoft/Phi-4-mini-instruct",trust_remote_code=True)
        self.index=None
        self.chunks=[]

    def extract_text_from_pdf(self,file_path):
        """
            Purpose: Extract text from pdfs
            Input: File Path
            Output: Text from the documents
        """
        try:
            text=""
            with fitz.open(file_path) as doc:
                for page in doc:
                    text+=page.get_text()
            logging.info("Text extracted from pdf")
            return text
        except Exception as e:
            logging.error("Error in extracting text from pdf "+e)
            return "Error in extracting text from pdf "+e

    def chunk_text(self,text, chunk_size=500,overlap=50):
        """
            Purpose: Split text into chunks
            Input: Text, chunk size and overlap
            Output: Return chunks of text
        """
        try:
            words=text.split()
            chunks=[]
            for i in range(0,len(words),chunk_size-overlap):
                chunk=" ".join(words[i:i+chunk_size])
                chunks.append(chunk)
            logging.info("Text split into chunks")
            return chunks
        except Exception as e:
            logging.error("Error in chunking of text "+e)
            return []

    def build_faiss_index_from_pdfs(self,uploaded_files):
        """
            Purpose: Create Faiss Index from pdfs
            Input: Uploaded files
            Output: Returns the Faiss index and chunks
        """
        try:
            logging.info("Carry out uploaded files")
            for file in uploaded_files:
                text=self.extract_text_from_pdf(file.name)
                file_chunks=self.chunk_text(text)
                self.chunks.extend(file_chunks)
            logging.info("Carry out embeddings")
            doc_embeddings=self.embedder.encode(self.chunks,convert_to_numpy=True)
            dimension=doc_embeddings.shape[1]
            logging.info("Create Faiss index")
            self.index=faiss.IndexFlatL2(dimension)
            self.index.add(doc_embeddings)
            return self.index, self.chunks
        except Exception as e:
            logging.error("Error in building faiss index from pdfs "+e)
            return self.index, self.chunks


    def get_response(self,query):
        """
            Purpose: Get response for the LLM for the query asked
            Input: Text of Query
            Output: Response generated for the text
        """
        try:
            if self.index is not None and self.chunks !=[]:
                logging.info("Converting the query to embeddings")
                query_embedding=self.embedder.encode([query],convert_to_numpy=True)
                D,I=self.index.search(query_embedding,k=2)
                retrieved_chunks=[self.chunks[i] for i in I[0]]
                logging.info("Relevant chunks for the query fetched")

                #Create prompt
                prompt=f"Context:\n{retrieved_chunks[0]}\n{retrieved_chunks[1]}\n\nQuestion:{query}\nAnswer:"

                #Generate response
                logging.info("Generate the response for the query")
                output=self.llm(prompt,max_new_tokens=50,do_sample=True, temperature=0.1)
                response=output[0]["generated_text"].split("Answer:")[-1].strip()
                logging.info("Response for the query generated")
            else:
                response="No reference data available"
            return response
        except Exception as e:
            logging.error("Error in exception "+e)
            return "Error in exception "+e



