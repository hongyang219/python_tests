import os
import io
import sys
import subprocess
import time

from typing_inspection.introspection import UNKNOWN

from agent_config import *
from python_examples import *


agent, MODEL = initialize_agent(llm_type="local")
agent_ol, MODEL_ol = initialize_agent(llm_type="online")

### Prompts area
system_message = '''
You are an assistant that reimplements Python code in high performance C++ for an M1 Mac.
Respond only with C++ code; use comments sparingly and do not provide any explanation other than occasional comments.
The C++ response needs to produce an identical output in the fastest possible time.
'''

def user_prompt_for(python):
    user_prompt = '''
    Rewrite this Python code in C++ with the fastest possible implementation that produces identical output in the least time.
    Respond only with C++ code; do not explain your work other than a few comments.
    Pay attention to number types to ensure no int overflows. Remember to #include all necessary C++ packages such as iomanip.\n\n
    '''
    user_prompt += python
    return user_prompt

### Tools
def messages_for(python):
    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_prompt_for(python)}
    ]

def current_time():
    now = time.localtime(time.time())
    return time.strftime('%m%d%H%M%S', now)

def write_output(cpp, code_name: str = 'PYtoCPP', output_dir: str = "./opt_c/"):
    code = cpp.replace("```cpp","").replace("```","")
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{code_name}OPT_{MODEL[:-3]}_{current_time()}.cpp"
    filepath = os.path.join(output_dir, filename)
    # print(filepath)
    with open(filepath, "w") as f:
        f.write(code)

def optimize_python_codes(python, code_name: str):
    stream = agent.chat.completions.create(model=MODEL, messages=messages_for(python), stream=True)
    reply = ""
    for chunk in stream:
        fragment = chunk.choices[0].delta.content or ""
        reply += fragment
        print(fragment, end='', flush=True)
    write_output(reply, code_name=code_name)


# write_output("TestCode","TestName")

# EXECUTION AREA
py = max_sub_array
code_name = "max_sub_array"
# exec(py)
# optimize_python_codes(py, code_name)


