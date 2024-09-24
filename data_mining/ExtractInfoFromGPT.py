import openai
from openai import OpenAI
import os

API_KEY = 'API_KEY'
# Ensure the OPENAI_API_KEY environment variable is set
os.environ['OPENAI_API_KEY'] = API_KEY
client = OpenAI()


def extract_info_from_gpt(post_content, prefix_prompt):
    try:
        completion = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": prefix_prompt + post_content,
                },
            ],
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(e)
        return "Gpt Failed\nGpt Failed"
