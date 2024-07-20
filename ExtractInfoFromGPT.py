import json

import openai
from openai import OpenAI
import os

API_KEY = 'key'
# Ensure the OPENAI_API_KEY environment variable is set
os.environ['OPENAI_API_KEY'] = API_KEY
client = OpenAI()

# Load the configuration from JSON file using UTF-8 encoding
with open('config.json', 'r', encoding='utf-8') as file:
    config = json.load(file)
prefix_prompt = config['prefix_prompt']


def extract_info_from_gpt(posts):
    suffix_prompt = ''
    for i, post in enumerate(posts):
        suffix_prompt += '\n' + f'post {i}:' + '\n' + post

    completion = openai.chat.completions.create(
        model="gpt-4",
        messages=[
            {
                "role": "user",
                "content": prefix_prompt + suffix_prompt,
            },
        ],
    )
    return completion.choices[0].message.content.strip()
