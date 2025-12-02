# 🎬 IMDB Sentiment Analysis API

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)  
[![FastAPI](https://img.shields.io/badge/Framework-FastAPI-green)](https://fastapi.tiangolo.com/)  
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Transformers-yellow)](https://huggingface.co/docs/transformers/index)  
[![Machine Learning](https://img.shields.io/badge/ML-Python-red)](https://www.python.org/)  

---

## 📝 Project Overview

This project provides a **FastAPI-based REST API** for predicting the sentiment of IMDb movie reviews using a fine-tuned **DistilBERT** transformer model.  

The API allows developers and applications to send text reviews and receive **real-time predictions** of positive or negative sentiment along with a confidence score.  

This API serves as a **deployable backend** for production applications, enabling seamless integration with web interfaces, mobile apps, or other services.

---

## 🎯 Objectives

- Expose a **RESTful API** for sentiment prediction using transformer models.
- Support **real-time inference** for text input.
- Handle **validation, errors, and edge cases** gracefully.
- Enable easy integration into external systems like web apps or dashboards.
- Demonstrate best practices for serving NLP models in production.

---

## 🛠️ Technical Workflow

1. **FastAPI Server**
   - Provides endpoints for health check and sentiment prediction.
   - Handles request validation and response formatting.

2. **Model Loading**
   - Fine-tuned **DistilBERT** model loaded from **Hugging Face Hub** at startup.
   - Tokenizer and model are cached for efficient inference.

3. **Request Schema**
   - **POST `/predict`** accepts JSON with a single field: `text`.
   ```json
   {
     "text": "This movie was amazing and thrilling!"
   }
   
4. **Prediction Logic**

- Input text is tokenized using `AutoTokenizer`.
- Model inference with `AutoModelForSequenceClassification`.
- Softmax applied to output logits to compute confidence scores.
- Returns predicted sentiment (`Positive` or `Negative`) and confidence value.

5. **API Endpoints**

- **GET /**: Health check endpoint to verify server is running.
- **POST /predict**: Accepts movie review text and returns predicted sentiment.

6. **Model Details**

- ### Phase 1: Transfer Learning (Classification Head)
  - Trains only the classification head for 2 epochs.
  - Validation Accuracy started at 80.68% and reached 80.94%.
  - Training loss decreased from 0.4741 to 0.4333.

- ### Phase 2: Fine-Tuning (Last Transformer Layers)
  - Unfroze the last 3 transformer layers (48 sub-layers) for 3 epochs.
  - Training Accuracy improved from 84.70% to 90.42%.
  - Validation Accuracy peaked at 88.16%.
  - Training Loss decreased from 0.3518 to 0.2320.

- ### Final Test Performance
  - Accuracy: 88.16%
  - F1-Score: 0.8809
  - Balanced performance across positive and negative classes.


## 🔹 **Recommendations**

- Expand Dataset: Fine-tune on additional datasets (Rotten Tomatoes, Amazon Reviews) for better generalization.

- Batch Predictions: Add batch processing endpoint for multiple reviews at once.

- Monitoring: Track request logs and prediction outcomes in production.

- Explainability: Integrate LIME or SHAP to understand predictions and increase trust.

- Caching: Consider caching repeated requests to reduce latency and compute cost.

- Security: Implement rate-limiting and authentication for API endpoints.

- Hyperparameter Tuning: Experiment with learning rate, batch size, and sequence length for potential performance gains.

## 🏁  **Conclusion**

This API demonstrates the deployment of a state-of-the-art transformer model for real-time sentiment analysis:

- Provides a reliable and scalable backend for NLP applications.

- Handles validation and error reporting to ensure robust integration.

- Enables developers to integrate sentiment analysis into web apps, dashboards, or mobile applications.

- Supports the practical use of transfer learning and transformer models in production systems.

- Serves as a template for end-to-end NLP deployment, from model fine-tuning to serving via a FastAPI endpoint.
  

## 🚀 How to Run This API Locally

### **Prerequisites**

- Python 3.9+

- FastAPI

- GPU recommended for faster inference (optional)

- Internet connection to download models from Hugging Face Hub

### **Installation**

1. Clone the repository:
  git clone <repository-url>
  cd <repository-folder>

2. Install dependencies
   pip install -r requirements.txt
  # Or manually:
  pip install fastapi uvicorn transformers torch huggingface_hub pydantic

3. Run the FastAPI server
   uvicorn main:app --reload
  NB: (Replace main.py with your script name if different)

4. Open API documentation in your browser
   http://127.0.0.1:8000/docs
   NB: Use the Swagger UI to test endpoints interactively.






