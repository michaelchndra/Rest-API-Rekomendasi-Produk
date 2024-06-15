# train.py
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import torch
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

# Load dataset
df = pd.read_csv('dataset.csv')

# Preprocessing
le = LabelEncoder()
df['label'] = le.fit_transform(df['rating_product'])
train_texts, val_texts, train_labels, val_labels = train_test_split(df['nama_product'], df['label'], test_size=0.2, random_state=42)

# Load pre-trained BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

# Tokenization
train_encodings = tokenizer(train_texts.tolist(), truncation=True, padding=True)
val_encodings = tokenizer(val_texts.tolist(), truncation=True, padding=True)

# Create PyTorch datasets
train_dataset = TensorDataset(
    torch.tensor(train_encodings['input_ids']),
    torch.tensor(train_encodings['attention_mask']),
    torch.tensor(train_labels.tolist())
)
val_dataset = TensorDataset(
    torch.tensor(val_encodings['input_ids']),
    torch.tensor(val_encodings['attention_mask']),
    torch.tensor(val_labels.tolist())
)

# Model setup
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=len(le.classes_),
    output_attentions=False,
    output_hidden_states=False
)

# Dataloader setup
train_loader = DataLoader(
    train_dataset,
    sampler=RandomSampler(train_dataset),
    batch_size=8
)
val_loader = DataLoader(
    val_dataset,
    sampler=SequentialSampler(val_dataset),
    batch_size=8
)

# Training function
def train_model(model, train_loader, val_loader, epochs=2):
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, eps=1e-8)
    total_steps = len(train_loader) * epochs
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
    criterion = torch.nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        for batch in tqdm(train_loader, desc="Training"):
            input_ids = batch[0]
            attention_mask = batch[1]
            labels = batch[2]
            
            model.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_train_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        
        avg_train_loss = total_train_loss / len(train_loader)
        print(f"Epoch {epoch + 1} - Average training loss: {avg_train_loss}")

        model.eval()
        total_eval_accuracy = 0
        total_eval_loss = 0
        predictions, true_labels = [], []
        for batch in tqdm(val_loader, desc="Validation"):
            input_ids = batch[0]
            attention_mask = batch[1]
            labels = batch[2]
            
            with torch.no_grad():
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits

            total_eval_loss += loss.item()
            logits = logits.detach().cpu().numpy()
            label_ids = labels.to('cpu').numpy()
            predictions.extend(np.argmax(logits, axis=1).flatten())
            true_labels.extend(label_ids.flatten())
        
        avg_val_loss = total_eval_loss / len(val_loader)
        print(f"Epoch {epoch + 1} - Validation Loss: {avg_val_loss}")
        print(f"Epoch {epoch + 1} - Validation Accuracy: {accuracy_score(true_labels, predictions)}")

        scheduler.step()

# Train model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
train_model(model, train_loader, val_loader, epochs=2)

# Save model
model_path = "bert_model.pth"
torch.save(model.state_dict(), model_path)
