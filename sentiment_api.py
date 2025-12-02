from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from huggingface_hub import hf_hub_download
import torch
import os


# FastAPI app
 
app = FastAPI(
    title="IMDB Sentiment Analysis API",
    description="Predict sentiment (positive/negative) for movie reviews using DistilBERT",
    version="1.0"
)


# Request schema

class Review(BaseModel):
    text: str


# Load DistilBERT from HF Hub

def load_model():
    """Load fine-tuned DistilBERT model from Hugging Face Hub."""
    repo_id = "nnejere/fine_tuned_distilbert_imdb"
    try:
        tokenizer_path = hf_hub_download(repo_id=repo_id, filename="tokenizer.json")
        model_path = hf_hub_download(repo_id=repo_id, filename="model.safetensors")

        tokenizer = AutoTokenizer.from_pretrained(os.path.dirname(tokenizer_path))
        model = AutoModelForSequenceClassification.from_pretrained(os.path.dirname(model_path))

        return tokenizer, model
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}")

# Load model once at startup
tokenizer, model = load_model()


# Prediction function

def predict_sentiment(text: str):
    if not text.strip():
        raise ValueError("Input text cannot be empty.")
    
    try:
        inputs = tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            padding=True, 
            max_length=128
        )
        with torch.no_grad():
            outputs = model(**inputs)
            preds = torch.softmax(outputs.logits, dim=1)
            confidence, label_id = torch.max(preds, dim=1)
            label = "Positive" if label_id.item() == 1 else "Negative"
            return {
                "label": label,
                "confidence": float(confidence)
            }
    except Exception as e:
        raise RuntimeError(f"Error during prediction: {e}")


# API routes

@app.get("/")
def health_check():
    return {"message": "Sentiment Analysis API is running."}

@app.post("/predict")
def analyze_sentiment(review: Review):
    try:
        result = predict_sentiment(review.text)
        return result
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except RuntimeError as re:
        raise HTTPException(status_code=500, detail=str(re))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")



