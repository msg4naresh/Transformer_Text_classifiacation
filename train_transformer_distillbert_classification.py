import torch
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import torch.optim as optim
from tqdm import tqdm
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Function to read data from CSV files
def read_data(true_file, fake_file):
    try:
        
        true_df = pd.read_csv(true_file)
        fake_df = pd.read_csv(fake_file)
        logger.info("Data loaded successfully.")
        return true_df, fake_df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return None, None

# Function to label data
def label_data(true_df, fake_df):
    true_df['isFake'] = 0
    fake_df['isFake'] = 1
    return true_df, fake_df

# Function to shuffle and split data
def shuffle_and_split(true_df, fake_df, test_size=0.2,limit_samples=100):
    combined_df = pd.concat([true_df, fake_df], ignore_index=True)
    
    # Limiting the number of samples for quicker execution
    combined_df = combined_df.sample(n=limit_samples).reset_index(drop=True)
    
    combined_df['content'] = combined_df['title'] + ' ' + combined_df['text']
    combined_df = combined_df[['content', 'isFake']] 
    
    train_data, test_data = train_test_split(combined_df, test_size=test_size)
    logging.info('Shuffled and split the data into training and testing sets.')
    return train_data,test_data



# Function for Tokenization
def encode_data(df, tokenizer, max_length=256):
    return tokenizer.batch_encode_plus(
        df['content'].values,
        add_special_tokens=True,
        return_attention_mask=True,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors='pt'
    )

# Function for Model Training
def train_model(model, train_dataloader, epochs=3, lr=1e-5):
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    
    for epoch in tqdm(range(1, epochs + 1)):
        model.train()
        
        for batch in train_dataloader:
            batch_input_ids = batch[0]
            batch_attention_mask = batch[1]
            batch_labels = batch[2]
            
            model.zero_grad()
            
            outputs = model(batch_input_ids, attention_mask=batch_attention_mask, labels=batch_labels)
            loss = outputs.loss
            loss.backward()
            
            optimizer.step()
        
        logging.info(f"Epoch {epoch} completed.")




# Function for Model Evaluation
def evaluate_model(model, test_dataloader, metrics=['accuracy', 'f1']):
    model.eval()
    all_preds = []
    all_labels = []
    
    for batch in test_dataloader:
        batch_input_ids = batch[0]
        batch_attention_mask = batch[1]
        batch_labels = batch[2]
        
        with torch.no_grad():
            outputs = model(batch_input_ids, attention_mask=batch_attention_mask)
        
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=1)
        
        # Storing all predictions and labels for later use
        all_preds.extend(predictions.cpu().numpy())
        all_labels.extend(batch_labels.cpu().numpy())
    if 'accuracy' in metrics:
        acc = accuracy_score(all_labels, all_preds)
        logging.info(f"Test Accuracy: {acc:.4f}")

    if 'f1' in metrics:
        f1 = f1_score(all_labels, all_preds, average='binary') 
        logging.info(f"Test F1 Score: {f1:.4f}")


def save_model(model, path='trained_model.pth'):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    logging.info(f"Model Saved Successfully")

if __name__ == "__main__":
    true_file = 'data/True.csv'
    fake_file = 'data/Fake.csv'
    
    true_df, fake_df = read_data(true_file, fake_file)
    true_df, fake_df = label_data(true_df, fake_df)
    train_df, test_df = shuffle_and_split(true_df, fake_df)
    
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    
    train_encoded = encode_data(train_df, tokenizer)
    logger.info("train data encoded successfully.")
    test_encoded = encode_data(test_df, tokenizer)
    logger.info("test data encoded successfully.")
    
    model = DistilBertForSequenceClassification.from_pretrained(
        'distilbert-base-uncased',
        num_labels=2,
        output_attentions=False,
        output_hidden_states=False
    )
    
    batch_size = 4
    train_dataset = TensorDataset(train_encoded['input_ids'], train_encoded['attention_mask'], torch.tensor(train_df['isFake'].values))
    train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=batch_size)
    logger.info("created Tensor dataset and dataloader for train data successfully.")
    
    test_dataset = TensorDataset(test_encoded['input_ids'], test_encoded['attention_mask'], torch.tensor(test_df['isFake'].values))
    test_dataloader = DataLoader(test_dataset, sampler=SequentialSampler(test_dataset), batch_size=batch_size)
    logger.info("created Tensor dataset and dataloader for test data successfully.")
    
    train_model(model, train_dataloader)
    logger.info("Model trained successfully.")
    evaluate_model(model, test_dataloader)
    logger.info("Model evaluation done successfully.")
    save_model(model, 'models/trained_model.pth')