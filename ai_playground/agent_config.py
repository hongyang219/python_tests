import os
from openai import OpenAI
from dotenv import load_dotenv
from huggingface_hub import login, InferenceClient
from transformers import AutoTokenizer


# Online models
deepseek_url = "https://api.deepseek.com"
DEEPSEEK = "deepseek-reasoner"

# Local models
ollama_url = "http://localhost:11434/v1"
QWEN = "qwen2.5-coder:7b"
DEEPSEEK_CODER = "deepseek-coder"
DEEPSEEK_R1 = "deepseek-r1:7b"

# Hugging Face pretrained models
QWEN_PT_URL = "https://router.huggingface.co/v1"
QWEN_PT = "Qwen/Qwen2.5-Coder-7B-Instruct:nscale"


# API Key
load_dotenv(override=True)
API_KEY = os.getenv('DEEPSEEK_API_KEY')
HF_TOKEN = os.getenv("HF_TOKEN")

ONLINE_MODEL = DEEPSEEK
LOCAL_MODEL = QWEN
PRETRAINED_MODEL = QWEN_PT


def initialize_agent(llm_type :str = "local"):
    if llm_type == "online":
        agent = OpenAI(
            api_key=API_KEY,
            base_url=deepseek_url,
        )
        model = ONLINE_MODEL
    elif llm_type == "local":
        agent = OpenAI(
            api_key="ollama",  # 本地部署不需要真实 Key，随便填一个字符串即可
            base_url=ollama_url  # Ollama 的默认本地地址
        )
        model = LOCAL_MODEL
    elif llm_type == "pretrained":
        agent = OpenAI(
            api_key=HF_TOKEN,  # 本地部署不需要真实 Key，随便填一个字符串即可
            base_url=ollama_url  # Ollama 的默认本地地址
        )
        model = PRETRAINED_MODEL
    else:
        print("Please enter either online or local or pretrained")
        exit()
    return agent, model


# Test Area
sys_prompt = '''
- 你是一个博物学家，通晓古今中外的历史、地理、科学、哲学、文学、心理学等学科的知识以及与之相关的名人轶事。
- 回答限制在300字以内
'''

usr_prompt = '''
- 斯多葛学派的主导思想是什么？
'''

def messages_for_testing():
    return [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": usr_prompt}
    ]

agent_test, MODEL_test = initialize_agent("local")

def play_model():
    stream = agent_test.chat.completions.create(model=MODEL_test, messages=messages_for_testing(), stream=True)
    reply = ""
    for chunk in stream:
        fragment = chunk.choices[0].delta.content or ""
        reply += fragment
        print(fragment, end='', flush=True)

# play_model()