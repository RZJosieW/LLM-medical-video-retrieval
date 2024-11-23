import json
import os
import torch
from transformers import BertTokenizer, BertForQuestionAnswering
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Step 1: Define paths and directories
train_file = 'train.json'
val_file = 'ckrval.json'  # Validation set file
subtitle_file = 'newsub.json'  # Path to new subtitle data file
model_path = '/mnt/bert/bert-large-uncased-whole-word-masking'  # Path to BERT Large model
output_dir = '/mnt/trained_model'  # Directory to save the trained model
weight_path = os.path.join(output_dir, 'bert_qa_model.pt')  # Path to save model weights

# Step 2: Load train data from JSON
with open(train_file, 'r', encoding='utf-8') as f:
    train_data = json.load(f)

# Step 3: Load validation data from JSON
with open(val_file, 'r', encoding='utf-8') as f:
    val_data = json.load(f)

# Step 4: Load subtitle data from newsub.json
with open(subtitle_file, 'r', encoding='utf-8') as f:
    subtitle_data = json.load(f)

# Step 5: Adjust subtitle_data structure for easier access
subtitle_data_dict = {item['video_id']: item['transcript'] for item in subtitle_data}

valid_examples = len(subtitle_data_dict)
print(f"Loaded {valid_examples} valid examples from newsub.json.")

# Step 6: Define a Dataset class for BERT input format
class QADataset(Dataset):
    def __init__(self, tokenizer, data, subtitle_data_dict, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.subtitle_data_dict = subtitle_data_dict
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]
        question = example['question']
        video_id = example['video_id']
        subtitles = self.subtitle_data_dict.get(video_id, [])

        # Combine subtitles into a single transcript
        subtitle_text = ' '.join([subtitle['text'] for subtitle in subtitles])

        # Tokenize question and subtitles
        question_tokens = self.tokenizer.encode(question, add_special_tokens=False)
        subtitle_tokens = self.tokenizer.encode(subtitle_text, add_special_tokens=False)

        # Truncate if combined length exceeds max_length - 3 (account for [CLS], [SEP], [SEP])
        combined_length = len(question_tokens) + len(subtitle_tokens) + 3
        if combined_length > self.max_length:
            subtitle_tokens = subtitle_tokens[:self.max_length - 3 - len(question_tokens)]

        # Create input_ids and attention_mask
        input_ids = [self.tokenizer.cls_token_id] + question_tokens + [self.tokenizer.sep_token_id] + subtitle_tokens + [self.tokenizer.sep_token_id]
        attention_mask = [1] * len(input_ids)

        # Padding
        padding_length = self.max_length - len(input_ids)
        input_ids += [self.tokenizer.pad_token_id] * padding_length
        attention_mask += [0] * padding_length

        # Convert lists to tensors
        input_ids = torch.tensor(input_ids)
        attention_mask = torch.tensor(attention_mask)
        start_positions = torch.tensor([example['answer_start_second']])
        end_positions = torch.tensor([example['answer_end_second']])

        return input_ids, attention_mask, start_positions, end_positions

# Step 7: Initialize tokenizer and model
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForQuestionAnswering.from_pretrained(model_path)

# Step 8: Prepare datasets and dataloaders
train_dataset = QADataset(tokenizer, train_data, subtitle_data_dict)
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)

val_dataset = QADataset(tokenizer, val_data, subtitle_data_dict)
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)

# Step 9: Define optimizer and training parameters
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
num_epochs = 3

# Step 10: Training loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.train()

for epoch in range(num_epochs):
    epoch_loss = 0.0
    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}', leave=False)
    for batch_idx, (input_ids, attention_mask, start_positions, end_positions) in enumerate(progress_bar):
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        start_positions = start_positions.to(device)
        end_positions = end_positions.to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, start_positions=start_positions, end_positions=end_positions)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        progress_bar.set_postfix({'batch_loss': loss.item(), 'epoch_loss': epoch_loss / (batch_idx + 1)})

    print(f"Epoch {epoch + 1}, Loss: {epoch_loss / len(train_loader)}")

    # Evaluation on validation set
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch_idx, (input_ids, attention_mask, start_positions, end_positions) in enumerate(val_loader):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            start_positions = start_positions.to(device)
            end_positions = end_positions.to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, start_positions=start_positions, end_positions=end_positions)
            val_loss += outputs.loss.item()

    avg_val_loss = val_loss / len(val_loader)
    print(f"Validation Loss after epoch {epoch + 1}: {avg_val_loss}")
    model.train()

# Save the trained model, tokenizer, and model weights
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
torch.save(model.state_dict(), weight_path)

print("Training completed and model saved.")
