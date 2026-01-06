# AI Playground

这是一个AI相关的Python项目游乐场，包含多种功能模块，用于测试和演示AI模型、代码转换等。

## 目录结构

- `agent_config.py`: 配置AI代理，支持在线、本地和预训练模型。使用OpenAI API接口，支持DeepSeek、Ollama等模型。
- `langchain.py`: 使用LangChain加载文档，从子文件夹中读取Markdown文件，并设置文档类型元数据。
- `pretrained_models.py`: 使用Hugging Face预训练模型，如CodeQwen和CodeGemma，进行代码生成和推理。
- `python_examples.py`: 包含Python代码示例，包括计算Pi、最大子数组和、判断幂等算法。
- `python_to_cpp.py`: 使用LLM将Python代码转换为高性能C++代码，并保存到opt_c目录。
- `ui_gradio.py`: 使用Gradio创建的Web界面，用于Python到C++代码转换，支持本地和在线LLM。
- `.env`: 环境变量文件，包含API密钥（DEEPSEEK_API_KEY, HF_TOKEN等）。
- `opt_c/`: 子目录，存放优化后的C++代码文件。

## 功能

- **AI模型配置**: 支持多种AI模型的初始化和调用。
- **文档加载**: 使用LangChain处理文档。
- **预训练模型推理**: 通过Hugging Face接口使用预训练模型。
- **代码示例**: 提供算法示例。
- **代码转换**: 将Python代码自动转换为C++。
- **用户界面**: Gradio Web UI用于代码转换。

## 使用方法

1. 配置.env文件中的API密钥。
2. 运行相应脚本测试功能。
3. 使用ui_gradio.py启动Web界面进行代码转换。

## 依赖

- openai
- dotenv
- huggingface_hub
- transformers
- langchain
- gradio

