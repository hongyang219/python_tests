import os
import io
import sys
import json
import requests
from dotenv import load_dotenv
from openai import OpenAI
from IPython.display import Markdown, display, update_display
import gradio as gr
import subprocess
from huggingface_hub import login, InferenceClient
from transformers import AutoTokenizer
from python_to_cpp import *

hf_token = os.getenv("HF_TOKEN")
login(hf_token, add_to_git_credential=True)

code_qwen = "Qwen/CodeQwen1.5-7B-Chat"
code_gemma = "google/codegemma-7b-it"
CODE_QWEN_URL = "https://h1vdol7jxhje3mpn.us-east-1.aws.endpoints.huggingface.cloud"
CODE_GEMMA_URL = "https://c5hggiyqachmgnqg.us-east-1.aws.endpoints.huggingface.cloud"


code_qwen_testerA = '''
tokenizer = AutoTokenizer.from_pretrained(code_qwen)
messages = messages_for(cal_pi)
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

print(text)

client = InferenceClient(CODE_QWEN_URL, token=hf_token)
stream = client.text_generation(text, stream=True, details=True, max_new_tokens=3000)
for r in stream:
    print(r.token.text, end = "")
'''

code_qwen_testerB = '''
client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=hf_token,
)

completion = client.chat.completions.create(
    model="Qwen/Qwen2.5-Coder-7B-Instruct:nscale",
    messages=[
        {
            "role": "user",
            "content": "What is the capital of France?"
        }
    ],
)
print(completion.choices[0].message)
'''
