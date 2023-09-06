import torch
import logging
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NewsClassifier:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            logger.info("Creating NewsClassifier instance and loading model.")
            cls._instance = super(NewsClassifier, cls).__new__(cls)
            # Load the tokenizer and model
            cls.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
            cls.model = DistilBertForSequenceClassification.from_pretrained(
                'distilbert-base-uncased',
                num_labels=2,
                output_attentions=False,
                output_hidden_states=False
            )
            # Load the trained model weights
            try:
                cls.model.load_state_dict(torch.load('models/trained_model.pth'))
                cls.model.eval()
                logger.info("Model loaded successfully.")
            except Exception as e:
                logger.error(f"Error loading model: {e}")
        return cls._instance

def predict_news(news: str) -> str:
    logger.info(f"Making a prediction for news: {news[:50]}...")  # Log the first 50 characters of the news
    classifier = NewsClassifier()
    tokenizer = classifier.tokenizer
    model = classifier.model

    try:
        inputs = tokenizer(news, return_tensors='pt', padding=True, truncation=True, max_length=256)
        with torch.no_grad():
            outputs = model(**inputs)
        
        logits = outputs.logits
        prediction = torch.argmax(logits, dim=1).item()
        logger.info(f"Prediction made: {'Fake' if prediction == 1 else 'True'}")
        
        return 'Fake' if prediction == 1 else 'True'
    except Exception as e:
        logger.error(f"An error occurred during prediction: {e}")
        return "Error during prediction"

