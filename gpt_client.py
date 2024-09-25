import openai
from openai import OpenAI
import os

def _get_api_key():
    with open('project_key.txt', 'r') as file:
        content = file.read()
    return content

api_key = _get_api_key()
openai.api_key = api_key

def query_chatgpt(query):
    global api_key
    os.environ['OPENAI_API_KEY'] = api_key
    client = OpenAI()

    completion = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "user",
            "content": query
        }
    ]
    )
    return completion.choices[0].message.content