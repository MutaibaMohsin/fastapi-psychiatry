import torch
import pandas as pd
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from fastapi import FastAPI
from pydantic import BaseModel
import nest_asyncio
import uvicorn
from pyngrok import ngrok

# 🔹 Initialize FastAPI
app = FastAPI()

# 🔹 Load AI Models
similarity_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
summarization_model = AutoModelForSeq2SeqLM.from_pretrained("google/long-t5-tglobal-base")
summarization_tokenizer = AutoTokenizer.from_pretrained("google/long-t5-tglobal-base")

# 🔹 Load Datasets (Upload CSV Files Manually)
recommendations_df = pd.read_csv("/content/drive/MyDrive/treatment_recommendations (2).csv")
questions_df = pd.read_csv("/content/drive/MyDrive/symptom_questions.csv")

# 🔹 Create FAISS Index for Treatment Retrieval
treatment_embeddings = similarity_model.encode(recommendations_df["Disorder"].tolist(), convert_to_numpy=True)
index = faiss.IndexFlatIP(treatment_embeddings.shape[1])
index.add(treatment_embeddings)

# 🔹 Create FAISS Index for Question Retrieval
question_embeddings = embedding_model.encode(questions_df["Questions"].tolist(), convert_to_numpy=True)
question_index = faiss.IndexFlatL2(question_embeddings.shape[1])
question_index.add(question_embeddings)

# 🔹 API Request Model
class ChatRequest(BaseModel):
    message: str

@app.post("/detect_disorders")
def detect_disorders(request: ChatRequest):
    """ Detect psychiatric disorders from user input """
    text_embedding = similarity_model.encode([request.message], convert_to_numpy=True)
    distances, indices = index.search(text_embedding, 3)
    disorders = [recommendations_df["Disorder"].iloc[i] for i in indices[0]]
    return {"disorders": disorders}

@app.post("/get_treatment")
def get_treatment(request: ChatRequest):
    """ Retrieve treatment recommendations """
    detected_disorders = detect_disorders(request)["disorders"]
    treatments = {disorder: recommendations_df[recommendations_df["Disorder"] == disorder]["Treatment Recommendation"].values[0] for disorder in detected_disorders}
    return {"treatments": treatments}

@app.post("/get_questions")
def get_recommended_questions(request: ChatRequest):
    """Retrieve the most relevant diagnostic questions based on patient symptoms."""
    input_embedding = embedding_model.encode([request.message], convert_to_numpy=True)
    distances, indices = question_index.search(input_embedding, 3)
    retrieved_questions = [questions_df["Questions"].iloc[i] for i in indices[0]]
    return {"questions": retrieved_questions}

@app.post("/summarize_chat")
def summarize_chat(request: ChatRequest):
    """ Summarize chat logs using LongT5 """
    inputs = summarization_tokenizer("summarize: " + request.message, return_tensors="pt", max_length=4096, truncation=True)
    summary_ids = summarization_model.generate(inputs.input_ids, max_length=500, num_beams=4, early_stopping=True)
    summary = summarization_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return {"summary": summary}

# # Allow FastAPI to run in Colab
# nest_asyncio.apply()
# !ngrok authtoken 2sipAHBxr0905blZoi7QiQ8SYER_6vi7NTQNiucisw7hqHtem
# # Expose API using ngrok
# public_url = ngrok.connect(8000).public_url
# print(f"🚀 Public API URL: {public_url}")

# # Start FastAPI Server
# uvicorn.run(app, host="0.0.0.0", port=8000)
