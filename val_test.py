import re
import torch
from transformers import BertTokenizer, BertForQuestionAnswering


tokenizer = BertTokenizer.from_pretrained('trained_model')


model_path = '/mnt/bert/trained_model/bert_qa_model.pt'
model = BertForQuestionAnswering.from_pretrained('trained_model')
model.load_state_dict(torch.load(model_path))


def parse_srt(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        srt_data = file.read()

    pattern = re.compile(r'(\d+\.\d+),(\d+\.\d+)\s+(.*?)\n', re.DOTALL)
    matches = pattern.findall(srt_data)

    subtitles = []
    for match in matches:
        start_time = float(match[0])
        end_time = float(match[1])
        text = match[2].strip()
        subtitles.append((start_time, end_time, text))

    return subtitles


srt_path = 'lbPbM8018CE.srt'
subtitles = parse_srt(srt_path)


text = " ".join([sub[2] for sub in subtitles])


question = "How to perform epley maneuver for vertigo?"


inputs = tokenizer(question, text, add_special_tokens=True, return_tensors='pt')
input_ids = inputs['input_ids'].tolist()[0]


outputs = model(**inputs)
answer_start_scores = outputs.start_logits
answer_end_scores = outputs.end_logits


answer_start = torch.argmax(answer_start_scores).item()
answer_end = torch.argmax(answer_end_scores).item() + 1


if answer_start >= len(subtitles):
    answer_start = len(subtitles) - 1
if answer_end >= len(subtitles):
    answer_end = len(subtitles) - 1


answer_start_second = subtitles[answer_start][0]
answer_end_second = subtitles[answer_end][1]


relevant_score = (answer_start_scores[0][answer_start] + answer_end_scores[0][answer_end - 1]).item()


result = {
    "video_id": "lbPbM8018CE",
    "relevant_score": relevant_score,
    "answer_start_second": answer_start_second,
    "answer_end_second": answer_end_second
}

print(result)
