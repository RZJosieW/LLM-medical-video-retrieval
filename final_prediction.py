import json
import os
import torch
import re
from transformers import T5Tokenizer, T5ForConditionalGeneration
from torch.utils.data import Dataset, DataLoader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer, BertModel
from torch import nn
import numpy as np
import math


class QuestionToVideoModel(nn.Module):
    def __init__(self, num_videos):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.video_id_classifier = nn.Linear(self.bert.config.hidden_size, num_videos)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        video_id_logits = self.video_id_classifier(pooled_output)
        return video_id_logits


def load_video_mapping(file_path):
    with open(file_path, 'r') as file:
        mapping = json.load(file)
    return mapping


def predict_question(model, tokenizer, question, device, video_id_to_idx):
    encoding = tokenizer.encode_plus(
        question,
        add_special_tokens=True,
        max_length=512,
        return_attention_mask=True,
        return_tensors='pt',
        truncation=True
    )

    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    model.eval()
    with torch.no_grad():
        video_id_logits = model(input_ids, attention_mask)

    predicted_video_id_index = torch.argmax(video_id_logits, dim=1).item()
    predicted_video_id = list(video_id_to_idx.keys())[list(video_id_to_idx.values()).index(predicted_video_id_index)]

    return predicted_video_id


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
mapping = load_video_mapping('videoid/video_id_mapping.json')
num_videos = len(mapping) 
model = QuestionToVideoModel(num_videos=num_videos).to(device)
model.load_state_dict(torch.load('model_epoch_20.pth', map_location=device))  
question = input("Please enter your question: ")
video_id = predict_question(model, tokenizer, question, device, mapping)
print(f"Predicted Video ID: {video_id}")

# Define file paths and directories
train_file = 'timestamp/train.json'
subtitle_dir = 'timestamp/subtitles'  # Modify to the actual subtitle folder name
local_model_path = 'timestamp/trained_model'  # Local model path
weight_path = os.path.join(local_model_path, 't5_qa_model.pt')  # Model weight save path

# Define the specific video ID and question
specific_video_id = predict_question(model, tokenizer, question, device, mapping)
fixed_question = question

# Load training data from JSON file
with open(train_file, 'r', encoding='utf-8') as f:
    train_data = json.load(f)


# Find the data for the specific video ID and question
def find_specific_data(video_id, question):
    for item in train_data:
        if item['video_id'] == video_id and item['question'] == question:
            return item
    return None


specific_data = find_specific_data(specific_video_id, fixed_question)

if not specific_data:
    print(f"Error: No data found for video_id {specific_video_id} with question '{fixed_question}'")
else:
    # Parse SRT file
    def parse_srt(file_path):
        if not os.path.exists(file_path):
            print(f"Warning: Subtitle file {file_path} does not exist!")
            return []

        with open(file_path, 'r', encoding='utf-8') as file:
            srt_data = file.read()

        # Define the pattern to match subtitles
        pattern = re.compile(r'{\s*(\d+\.\d+),\s*(\d+\.\d+)\s*(.*?)\s*}', re.DOTALL)
        matches = pattern.findall(srt_data)

        subtitles = []
        for match in matches:
            start_time = float(match[0])
            end_time = float(match[1])
            text = match[2].strip()
            subtitles.append((start_time, end_time, text))

        return subtitles


    def extract_subtitle_text(subtitles, start_time, end_time):
        """Extract text from subtitles based on given start and end times"""
        text = []
        for sub in subtitles:
            if sub[0] >= start_time and sub[1] <= end_time:
                text.append(sub[2])
        return ' '.join(text)


    def find_top_n_similar_subtitles(prediction, subtitles, top_n=3):
        """Find the top N subtitle segments with the highest similarity score to the prediction text"""
        subtitle_texts = [sub[2] for sub in subtitles]

        if not subtitle_texts:
            print("Warning: Subtitle text list is empty, cannot calculate similarity!")
            return []

        subtitle_texts.append(prediction)  # Include prediction text

        # Ensure all texts are strings
        subtitle_texts = [str(text) for text in subtitle_texts]

        # Use TF-IDF vectorizer with ngram_range
        vectorizer = TfidfVectorizer(ngram_range=(1, 3))
        tfidf_matrix = vectorizer.fit_transform(subtitle_texts)

        if tfidf_matrix.shape[0] <= 1:
            print("Warning: Insufficient samples in the TF-IDF matrix!")
            return []

        # Calculate cosine similarity
        cosine_sim = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])

        # Get similarity scores and their corresponding indices
        similarity_scores = cosine_sim[0]
        indexed_scores = list(enumerate(similarity_scores))
        indexed_scores.sort(key=lambda x: x[1], reverse=True)  # Sort by similarity

        # Select the top N subtitle segments with the highest similarity scores
        top_n_indices = [index for index, score in indexed_scores[:top_n]]
        top_n_subtitles = [subtitles[index] for index in top_n_indices]

        return top_n_subtitles


    # Define a Dataset class for T5 input format
    class SingleVideoQADataset(Dataset):
        def __init__(self, tokenizer, question, video_id, subtitle_dir, max_length=1024):
            self.question = question
            self.video_id = video_id
            self.tokenizer = tokenizer
            self.subtitle_dir = subtitle_dir
            self.max_length = max_length

        def __len__(self):
            return 1  # Only one item

        def __getitem__(self, idx):
            srt_path = os.path.join(self.subtitle_dir, f'{self.video_id}.srt')

            # Check if subtitle file exists
            if not os.path.exists(srt_path):
                print(f"Warning: Subtitle file {srt_path} does not exist!")
                return {
                    'input_ids': torch.zeros(self.max_length, dtype=torch.long),
                    'question': self.question,
                    'subtitles': []  # Empty subtitle list
                }

            # Parse SRT file
            subtitles = parse_srt(srt_path)
            subtitle_text = extract_subtitle_text(subtitles, 0, float('inf'))  # Extract all subtitles

            # Format as T5 model input
            input_text = f"question: {self.question} context: {subtitle_text}"

            # Encode the input
            input_ids = self.tokenizer.encode(input_text, truncation=True, padding='max_length',
                                              max_length=self.max_length,
                                              return_tensors='pt')

            return {
                'input_ids': input_ids.squeeze(),
                'question': self.question,
                'subtitles': subtitles  # Add subtitle information
            }


    # Initialize tokenizer and model
    tokenizer = T5Tokenizer.from_pretrained(local_model_path)
    model = T5ForConditionalGeneration.from_pretrained(local_model_path)

    # Load trained model weights
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()

    # Prepare dataset and dataloader
    test_dataset = SingleVideoQADataset(tokenizer, fixed_question, specific_video_id, subtitle_dir)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Prediction

    model.to(device)

    current_question = None
    current_prediction = []
    current_subtitles = []
    results = {}  # dictionary to store the outputs


    def format_time(seconds):
        """Convert seconds to minutes:seconds format"""
        minutes = int(seconds // 60)
        seconds = int(seconds % 60)
        return f"{minutes:02}:{seconds:02}"


    def count_sentences(text):
        """Count the number of sentences in the text"""
        sentence_endings = re.compile(r'[.!?]')
        return len(sentence_endings.findall(text))


    def calculate_iou(true_start, true_end, pred_start, pred_end):
        """Calculate the IOU between the true and predicted values"""
        start_max = max(true_start, pred_start)
        end_min = min(true_end, pred_end)
        intersection = max(0, end_min - start_max)
        union = max(true_end, pred_end) - min(true_start, pred_start)
        return intersection / union if union > 0 else 0


    # Track the total IOU and count
    iou_total = 0
    iou_count = 0


    def process_prediction(current_question, current_prediction, current_subtitles, results, item):
        if not current_question:
            return

        merged_prediction = " ".join(current_prediction).replace(' .', '.').strip()

        if not merged_prediction:
            print(f"Warning: Prediction result is empty, skipping question: {current_question}")
            return

        # Calculate the number of sentences in the prediction result
        num_sentences = count_sentences(merged_prediction)
        if num_sentences <= 4:
            two_thirds = 8
        else:
            two_thirds = (3 / 2) * num_sentences
        result = math.ceil(two_thirds)

        # Find the top N subtitle segments most similar to the predicted answer
        top_n_subtitles = find_top_n_similar_subtitles(merged_prediction, current_subtitles,
                                                       top_n=result or 3)  # Select at least 3

        if top_n_subtitles:
            # Sort the timestamps of the selected subtitle segments
            top_n_subtitles.sort(key=lambda sub: sub[0])  # Sort by start time
            sorted_by_end_time = sorted(top_n_subtitles, key=lambda sub: sub[1])  # Sort by end time

            # Merge content
            merged_content = ''
            try:
                # Ensure sub[2] is a subtitle content string
                merged_content = ' '.join(sub[2] for sub in top_n_subtitles if isinstance(sub[2], str))
            except TypeError as e:
                print(f"Error: The elements in top_n_subtitles are not as expected: {e}")
                print("top_n_subtitles:", top_n_subtitles)
                merged_content = 'Error processing subtitles.'

            # Calculate start and end times for prediction
            if result >= 18:
                if len(sorted_by_end_time) >= 2:
                    end_time = float(sorted_by_end_time[-8][1])  # Second-largest end time
                    start_time = float(sorted_by_end_time[4][0])  # Second-smallest start time

            elif 18 > result >= 12:
                if len(sorted_by_end_time) >= 2:
                    end_time = float(sorted_by_end_time[-4][1])  # Fourth largest end time
                    start_time = float(sorted_by_end_time[4][0])  # Third-smallest start time

            elif 12 > result >= 8:
                if len(sorted_by_end_time) >= 2:
                    end_time = float(sorted_by_end_time[-3][1])  # Fourth largest end time
                    start_time = float(sorted_by_end_time[3][0])  # Third-smallest start time

            elif 8 > result > 3:
                if len(sorted_by_end_time) >= 2:
                    end_time = float(sorted_by_end_time[-2][1])  # Third-largest end time
                    start_time = float(sorted_by_end_time[1][0])  # Second-smallest start time

            else:  # 3 >= result
                if len(sorted_by_end_time) >= 2:
                    end_time = float(sorted_by_end_time[-1][1])  # Second-largest end time
                    start_time = float(sorted_by_end_time[1][0])  # Second-smallest start time

            # Extract and format times
            true_start = specific_data['answer_start_second']
            true_end = specific_data['answer_end_second']
            formatted_true_start = format_time(true_start)
            formatted_true_end = format_time(true_end)
            formatted_pred_start = format_time(start_time)
            formatted_pred_end = format_time(end_time)

            # Calculate IOU
            iou = calculate_iou(true_start, true_end, start_time, end_time)

            # Update total IOU and count
            global iou_total, iou_count
            iou_total += float(iou)
            iou_count += 1

            # Calculate current average IOU
            average_iou = iou_total / iou_count if iou_count > 0 else 0

            # Extract individual sentence timestamps
            sentence_timestamps = [(format_time(sub[0]), format_time(sub[1])) for sub in top_n_subtitles]

            # Print results
            print(f"Question: {current_question}")
            print(f"True Answer: {formatted_true_start} - {formatted_true_end}")
            print(f"Predicted Answer: {merged_prediction}")
            print(f"Number of sentences in prediction: {num_sentences} {result}")
            print(f"Most similar subtitle times: {', '.join([f'{{{t[0]},{t[1]}}}' for t in sentence_timestamps])}")
            print(f"IOU: {float(iou):.2f}")
            print(f"Current Average IOU: {average_iou:.2f}")

            # New: Print the determined answer start and end time
            print(f"Determined Answer Start Time: {formatted_pred_start}")
            print(f"Determined Answer End Time: {formatted_pred_end}")
            print("-" * 50)
            question_id = "Q" + str(item["sample_id"])
            results['question_id'] = question_id,
            results['relevant_video'] = {
                "video_id": item["video_id"],
                "relevant score": float(iou),
                "answer_start": f"{formatted_pred_start}",
                "answer_end": f"{formatted_pred_end}",
            }


    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        question = batch['question']
        subtitles = batch['subtitles']

        with torch.no_grad():
            preds = model.generate(input_ids,
                                   num_beams=5,  # Use beam search to increase diversity
                                   no_repeat_ngram_size=2,  # Prevent repeated n-grams
                                   max_length=250,  # Maximum length of generated text
                                   min_length=100,  # Minimum length of generated text
                                   length_penalty=1.0,  # Length penalty
                                   early_stopping=True)  # Early stopping

            decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

        if current_question is None:
            current_question = question
            current_subtitles = subtitles

        if question == current_question:
            # Append prediction results
            current_prediction.extend(decoded_preds)
        else:
            # Process predictions for the previous question
            process_prediction(current_question, current_prediction, current_subtitles, results, specific_data)

            # Reset current question and prediction results
            current_question = question
            current_prediction = decoded_preds
            current_subtitles = subtitles

    # Process the last question
    process_prediction(current_question, current_prediction, current_subtitles, results, specific_data)
    # Save results to JSON file
    with open("results.json", "w", encoding="utf-8") as json_file:
        json.dump(results, json_file, ensure_ascii=False, indent=4)
