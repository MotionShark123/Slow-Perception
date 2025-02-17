import re
import json


def process_value(value):
    return re.sub(r'%.*?\n', '', value)

def process_conversations(conversations):
    for conversation in conversations:
        if 'value' in conversation:
            conversation['value'] = process_value(conversation['value'])

def process_json(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as infile:
        data = json.load(infile)
    for item in data:
        if 'conversations' in item:
            process_conversations(item['conversations'])
    with open(output_file, 'w', encoding='utf-8') as outfile:
        json.dump(data, outfile, indent=4, ensure_ascii=False)