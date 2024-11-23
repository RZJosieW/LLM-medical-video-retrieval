import os
import re
import json
import numpy as np

def preprocess_transcript(transcript):
    segments = []
    for entry in transcript:
        start_time = entry['timestamp'][0]
        end_time = entry['timestamp'][1]
        text = entry['text']
        segments.append((start_time, end_time, text))
    return segments

def merge_segments(segments, max_length=512):
    merged_segments = []
    current_text = ""
    current_start = segments[0][0]
    current_end = segments[0][1]

    for start_time, end_time, text in segments:
        if len(current_text) + len(text) <= max_length:
            current_text += " " + text
            current_end = end_time
        else:
            merged_segments.append((current_start, current_end, current_text.strip()))
            current_text = text
            current_start = start_time
            current_end = end_time

    merged_segments.append((current_start, current_end, current_text.strip()))
    return merged_segments

def preprocess_data(data):
    processed_data = []
    unique_video_ids = set()
    for entry in data:
        video_id = entry['video_id']
        unique_video_ids.add(video_id)
        transcript = preprocess_transcript(entry['transcript'])
        merged_segments = merge_segments(transcript)
        for start_time, end_time, text in merged_segments:
            processed_data.append({
                'video_id': video_id,
                'start_time': start_time,
                'end_time': end_time,
                'text': text
            })
    return processed_data, list(unique_video_ids)

if __name__ == "__main__":
    with open('train_srt.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    processed_data, unique_video_ids = preprocess_data(data)

    with open('preprocessed_data.json', 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=4)

    with open('unique_video_ids.json', 'w', encoding='utf-8') as f:
        json.dump(unique_video_ids, f, ensure_ascii=False, indent=4)

    print("Preprocessing complete. Data saved to 'preprocessed_data.json' and 'unique_video_ids.json'.")
