import json
import torch
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle
import os

def preprocess_data(data_path):
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    texts = []
    combined_labels = []

    for item in data:
        video_id = item['video_id']
        start_time = item['start_time']
        end_time = item['end_time']
        combined_label = f"{video_id}_{start_time}_{end_time}"
        texts.append(item['text'])
        combined_labels.append(combined_label)

    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(combined_labels)

    return texts, labels, label_encoder

def save_label_encoder(label_encoder, filepath):
    with open(filepath, 'wb') as f:
        pickle.dump(label_encoder, f)

def train_model(train_texts, train_labels, val_texts, val_labels, model_dir, tokenizer_name, num_labels):
    tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
    model = BertForSequenceClassification.from_pretrained(tokenizer_name, num_labels=num_labels)

    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=512)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=512)

    class TextDataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item['labels'] = torch.tensor(self.labels[idx])
            return item

        def __len__(self):
            return len(self.labels)

    train_dataset = TextDataset(train_encodings, train_labels)
    val_dataset = TextDataset(val_encodings, val_labels)




    training_args = TrainingArguments(
        output_dir=model_dir,
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=os.path.join(model_dir, 'logs'),  # Save logs in ./trained_model/logs
        logging_steps=5000,  # Log every 5000 steps
        evaluation_strategy="epoch",  # Evaluate at the end of each epoch
        save_steps=5000,  # Save checkpoint every 5000 steps
    )



    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    trainer.train()
    trainer.save_model(model_dir)

def main():
    data_path = 'preprocessed_data.json'
    model_dir = './trained_model'
    tokenizer_name = 'bert-base-uncased'

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    texts, labels, label_encoder = preprocess_data(data_path)

    train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.1)

    num_labels = len(np.unique(labels))

    train_model(train_texts, train_labels, val_texts, val_labels, model_dir, tokenizer_name, num_labels)
    save_label_encoder(label_encoder, os.path.join(model_dir, 'combined_to_label.pkl'))

    print("Training complete. Model saved to", model_dir)

if __name__ == "__main__":
    main()
