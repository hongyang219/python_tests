import gradio as gr
from python_to_cpp import *

def stream_output(python):
    stream = agent.chat.completions.create(model=MODEL, messages=messages_for(python), stream=True)
    reply = ""
    for chunk in stream:
        fragment = chunk.choices[0].delta.content or ""
        reply += fragment
        yield reply.replace('```cpp\n','').replace('```','')

def stream_openai_local(python):
    stream = agent.chat.completions.create(model=MODEL, messages=messages_for(python), stream=True)
    reply = ""
    for chunk in stream:
        fragment = chunk.choices[0].delta.content or ""
        reply += fragment
        yield reply.replace('```cpp\n','').replace('```','')

def stream_openai_online(python):
    stream = agent_ol.chat.completions.create(model=MODEL_ol, messages=messages_for(python), stream=True)
    reply = ""
    for chunk in stream:
        fragment = chunk.choices[0].delta.content or ""
        reply += fragment
        yield reply.replace('```cpp\n','').replace('```','')

def optimize(python, model):
    if model=="Local LLM":
        result = stream_openai_local(python)
    elif model=="Online LLM":
        result = stream_openai_online(python)
    else:
        raise ValueError("Unknown model")
    for stream_so_far in result:
        yield stream_so_far

def execute_python(code):
    try:
        output = io.StringIO()
        sys.stdout = output
        exec(code)
    finally:
        sys.stdout = sys.__stdout__
    return output.getvalue()

def execute_cpp(code):
    # write_output(code, code_name='cal_pi')
    print("This function is not supported yet")
    # try:
        # compile_cmd = ["clang++", "-Ofast", "-std=c++17", "-march=armv8.5-a", "-mtune=apple-m1", "-mcpu=apple-m1",
        #                 "-o", "optimized", "optimized.cpp"]
        # compile_result = subprocess.run(compile_cmd, check=True, text=True, capture_output=True)
        # run_cmd = ["./optimized"]
        # run_result = subprocess.run(run_cmd, check=True, text=True, capture_output=True)
        # return run_result.stdout
    # except subprocess.CalledProcessError as e:
    #         return f"An error occurred:\n{e.stderr}"

css = """
.python {background-color: #306998;}
.cpp {background-color: #050;}
"""


# with gr.Blocks() as ui:
#     with gr.Row():
#         python = gr.Textbox(label="Python code:", lines=10, value=cal_pi)
#         cpp = gr.Textbox(label="C++ code:", lines=10)
#     with gr.Row():
#         model = gr.Dropdown(["Local LLM", "Online LLM"], label="Select model", value="Local LLM")
#         convert = gr.Button("Convert code")
#
#     convert.click(optimize, inputs=[python, model], outputs=[cpp])

with gr.Blocks(css=css) as ui:
    gr.Markdown("## Convert code from Python to C++")
    with gr.Row():
        python = gr.Textbox(label="Python code:", value=cal_pi, lines=10)
        cpp = gr.Textbox(label="C++ code:", lines=10)
    with gr.Row():
        model = gr.Dropdown(["Local LLM", "Online LLM"], label="Select model", value="Local LLM")
    with gr.Row():
        convert = gr.Button("Convert code")
    with gr.Row():
        python_run = gr.Button("Run Python")
        cpp_run = gr.Button("Run C++ (Not supported)")
    with gr.Row():
        python_out = gr.TextArea(label="Python result:", elem_classes=["python"])
        cpp_out = gr.TextArea(label="C++ result:", elem_classes=["cpp"])

    convert.click(optimize, inputs=[python, model], outputs=[cpp])
    python_run.click(execute_python, inputs=[python], outputs=[python_out])
    cpp_run.click(execute_cpp, inputs=[cpp], outputs=[cpp_out])

ui.launch(inbrowser=True)