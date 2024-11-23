import json
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.preprocessing import LabelEncoder
import pickle

def load_label_encoder(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)

def predict_and_evaluate(test_data, model_dir, tokenizer_name, label_encoder):
    tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
    model = BertForSequenceClassification.from_pretrained(model_dir)
    model.eval()

    predictions = []
    for item in test_data:
        if 'question' not in item:
            raise KeyError(f"Missing 'question' in data item: {item}")

        inputs = tokenizer(item["question"], padding='max_length', truncation=True, max_length=512, return_tensors='pt')
        inputs = {key: val.to(model.device) for key, val in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            predicted_label_id = logits.argmax(-1).item()

            try:
                predicted_combined_label = label_encoder.inverse_transform([predicted_label_id])[0]
                video_id, start_time, end_time = predicted_combined_label.split('_')
                start_time = float(start_time)
                end_time = float(end_time)
            except ValueError as e:
                print(f"Warning: Failed to inverse transform label_id {predicted_label_id}. Setting default values.")
                video_id, start_time, end_time = "unknown", 0.0, 0.0  

        predictions.append({
            "sample_id": item["sample_id"],
            "question": item["question"],
            "actual_video_id": item["video_id"],
            "predicted_video_id": video_id,
            "actual_answer_start_second": item["answer_start_second"],
            "predicted_answer_start_second": start_time,
            "actual_answer_end_second": item["answer_end_second"],
            "predicted_answer_end_second": end_time
        })

    return predictions

def main():
    test_data_path = 'test.json'  
    model_dir = './trained_model/checkpoint-15663'
    tokenizer_name = 'bert-base-uncased'

    label_encoder = load_label_encoder('./trained_model/combined_to_label.pkl')  

    with open(test_data_path, 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    predictions = predict_and_evaluate(test_data, model_dir, tokenizer_name, label_encoder)

    with open('predictions.json', 'w', encoding='utf-8') as f:
        json.dump(predictions, f, ensure_ascii=False, indent=4)

    print("Predictions saved to predictions.json")

if __name__ == "__main__":
    main()
