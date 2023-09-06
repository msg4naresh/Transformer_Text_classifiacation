
from fastapi import FastAPI
from pydantic import BaseModel
from news_classifier import predict_news

app = FastAPI()

class NewsItem(BaseModel):
    news: str

@app.get("/")
def read_root():
    return {"Text Classifier": "FakeNews Classifaiction Using Transformer"}

@app.post("/predict/")
async def classify_news(news_item: NewsItem):
    label = predict_news(news_item.news)
    return {"prediction": label}