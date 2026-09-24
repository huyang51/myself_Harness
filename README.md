# myself_Harness
> 从0到1手搓属于自己的harness

# DAY 1
今天首先通过代码构建与大模型的连接，有四种不同的连接方式.

## 1.1原生HTTP请求方式
使用python的'requests'库直接发送http请求
采用openai的兼容的API格式
### （1）优势和场景
**优势：**

- 灵活性最高，可以完全自定义请求参数

- 不依赖特定SDK，减少依赖

- 适合需要精确控制请求细节的场景

**适用场景：**

- 自定义LLM服务调用

- 需要特殊请求头或认证方式

- 对性能要求较高的生产环境
### （2）代码示例

```python
import requests
import json
#远程服务的api端点
url = 'https://api.deepseek.com/chat/completions'

#请求头
headers = {
    'Content-Type':"application/json",#声明请求体的格式是json
    "Authorization":'Bearer <API-KEY>'#api-key认证
}

#请求体
data = {
    "model":"deepseek-flash",
    "messages":[
        {'role':'user','content':'你是什么模型？'}
    ],
    "max_tokens":2048,
    "temperature":0.7,
    "top_k":1,
    "top_p":0.75,
    "stream":True,

}

#发送post请求
response = requests.post(url,headers=headers,data=json.dumps(data))

# 解析并打印回复内容
if response.status_code == 200:
    #流式解析
    print(response)
    for line in response.iter_lines(decode_unicode=True):
        if not line:
            continue
               
        if line.startswith("data:"):
            data_str = line[len("data:"):].strip()
           
            if data_str=='[DONE]':
                break

            chunks = json.loads(data_str)
            delta = chunks['choices'][0]['delta']
            content = delta.get('content') or ''
            print(content,end='')


else:
    print(f'请求失败，状态码：{response.status_code}')
    print(response.text)
```

### (3)参数解读
- messages:role
  messages 是一个数组，按时间顺序排列对话历史。role 字段定义了每条消息的发送者身份，模型会根据这个身份来理解上下文。
  **user**：代表终端用户。你代码中的 {'role':'user','content':'你是什么模型？'} 就是你作为用户向模型提出的问题。
  **assistant**：代表模型（AI）。在多轮对话中，你需要将模型之前的回复以 assistant 角色加入 messages 数组，这样模型才能记住上下文。

  **system**：代表系统级指令。用于设定模型的角色、行为规范或安全边界。例如：{"role": "system", "content": "你是一个乐于助人的助手。"}。

  **tool**：代表工具执行结果。当模型请求调用外部函数（工具）后，你需要将函数的返回结果以 tool 角色发送给模型，让模型基于结果继续生成回复
- **model**:指定要使用的模型ID名称
- **max_tokens**：限制模型最多生成的 token 数量。
- **temperature**：控制输出的随机性。取值范围通常在 0 到 2 之间。值越高（如 0.8），输出越随机、有创造性；值越低（如 0.2），输出越聚焦、确定。
- **top_p**：核采样参数，是另一种控制随机性的方法。模型只从概率质量累积达到 top_p 的 token 集合中采样。你设置为 0.75，意味着只考虑累积概率为 75% 的 token。通常建议 temperature 和 top_p 只调整其中一个，不要同时修改
- **top_k**：限制采样时只从概率最高的 k 个 token 中选择。
- **stream**：布尔值。设置为 true 时，响应会以流式方式返回，即生成一部分就立即发送一部分，而不是等全部生成完再返回。这对于需要实时显示回复的聊天界面非常有用
- **stop**：字符串或字符串数组。设置一个或多个停止序列。当模型生成的内容包含这些序列时，会立即停止生成
- **frequency_penalty**：数值，范围 -2.0 到 2.0。降低模型重复使用相同 token 的频率。值越高，模型越倾向于使用新词汇。目前已有部分api弃用。
- **presence_penalty**：数值，范围 -2.0 到 2.0。鼓励模型谈论新话题。值越高，模型越倾向于引入新的概念
- **tools**：一个数组，用于定义模型可以调用的外部函数（工具）。当你需要模型执行计算、查询数据库等操作时，可以通过此参数赋予模型相应的能力
- **tool_choice**：控制模型是否/如何调用 tools 中定义的函数。可选值有 none、auto、required 或指定特定函数名
- **response_format**：指定模型输出的格式。设置为 {"type": "json_object"} 可以强制模型输出合法的 JSON 对象。使用时，需要在 system 或 user 的提示词中明确要求输出 JSON，并最好给出格式示例
- logprobs 与 top_logprobs：用于调试和分析。设置 logprobs: true 后，响应中会包含每个生成 token 的对数概率信息，top_logprobs 还可以指定返回每个位置最可能的 k 个 token 及其概率
## 1.2 封装式API调用
将API调用逻辑封装成函数
使用completion API而非chat completion API
### (1)优势和适用场景
**优势：**

- 代码复用性好，便于维护

- 支持多模型切换

- 统一的错误处理机制

**适用场景：**

- 需要频繁切换不同模型的场景

- 批量处理多个请求

- 作为其他项目的依赖模块

### （2）代码
```python
import requests
import json
 
def llm_inference(prompt_list: list, model_name: str):
    if model_name == "qwen2.5_32b_awq":
        llm_server = {"server_url": "http://127.0.0.1:6790/v1/completions",
                      "path": "/models/Qwen2___5-32B-Instruct-AWQ"}
 
    elif model_name == "qwen2.5_7b_awq":
        llm_server = {"server_url": "http://127.0.0.1:6791/v1/completions",
                      "path": "/models/Qwen2___5-7B-Instruct-AWQ"}
    else:
        llm_server = {"server_url": "http://127.0.0.1:6790/v1/completions",
                      "path": "/models/Qwen2___5-32B-Instruct-AWQ"}
 
    # system_text = ""
    # prompt = f"<|im_start|>system\n{system_text}<|im_end|>\n<|im_start|>user\n{query_text}<|im_end|><|im_start|>assistant\n"
    rewrite_server_url = llm_server["server_url"]
    rewrite_server_headers = {
        'Content-Type': 'application/json'
    }
    rewrite_server_data = {
        'model': llm_server["path"],
        'prompt': prompt_list,
        'max_tokens': 4096, # 生成长度
        'top_k': 1,
        'top_p': 0.75,
        'temperature': 0,
        'stop': ["<|im_end|>"]
    }
    response = requests.post(rewrite_server_url, headers=rewrite_server_headers, data=json.dumps(rewrite_server_data))
    # return response.json()['choices'][0]['text']
    return response.json()
 
if __name__ == "__main__":
    prompt_list = [f"<|im_start|>system\n{'你是围城智能机器人'}<|im_end|>\n<|im_start|>user\n{'你是谁'}<|im_end|><|im_start|>assistant\n"]
 
    answer = llm_inference(prompt_list, "qwen2.5_32b_awq")
    # print(answer)
    for i in range(len(prompt_list)):
        print(answer['choices'][i]['text'])
```
## 1.3OpenAI SDK方式
使用官方的OpenAI Python SDK.
### （1）优势和适用场景
**优势：**

- 使用官方SDK，稳定性高

- 自动处理认证和请求格式

- 支持流式响应等高级功能

**适用场景：**

- 调用OpenAI官方或兼容API

- 需要SDK提供的便利功能

- 快速原型开发
### （2）代码
```python
from openai import OpenAI
import logging
logger = logging.Logger(__name__)
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler('prob.log',encoding='utf-8')
file_handler.setLevel(logging.DEBUG)

logger.addHandler(console_handler)
logger.addHandler(file_handler)



def send_test_request(client,model_name,test_message='你是什么模型'):
    #构建消息列表
    messages = [
        {'role':'user',"content":[{'type':"text","text":test_message}]
        }
    ]
    response = client.chat.completions.create(
        model=model_name,
        messages = messages,
        temperature=0.0
    )
    logger.info(f'Response:{response}')
    return response.content[0]


if __name__ == "__main__":
    openai_api_key = ''
    openai_api_base = ''
    model_name = ''
    client = OpenAI(api_key=openai_api_key,
                    base_url=openai_api_base)

    try:
        response = send_test_request(client=client,model_name=model_name)
        logger.info(response)
    except Exception as e:
        logger.error(e)
```

## 总结
第一次主要是熟悉python代码调用大模型对话的方式，这是第一步，后续也将逐步实现更多功能。


# DAY 2
昨天我们熟悉了通过python代码调用远程大模型的方式，今天我们学习一下下载部署本地小模型并调用的方式。

## 1.Ollama
Ollama 是目前最简单的本地大模型部署工具，一行命令安装，一行命令下载模型，一行命令开始对话。它支持 Windows、macOS、Linux 三大平台，提供 OpenAI 兼容 API，数据完全保留在本机。
### 1.1 安装Ollama
**Windows 用户**：访问 ollama.com 下载 OllamaSetup.exe，双击启动，点击 Install 即可。安装程序会自动安装到当前用户目录，无需手动配置路径。

**macOS 用户**：下载 Ollama.dmg 后双击打开，将 Ollama 拖入应用程序文件夹即可。

**Linux 用户**：使用官方一键安装脚本：
```bash
curl -fsSL https://ollama.com/install.sh | sh
```
安装完成后 Ollama 会自动启动，系统托盘出现 Ollama 图标即表示后台服务已在运行.

### 1.2修改模型存储路径
Ollama 默认将模型存储到 C 盘（C:\Users\Administrator\.ollama\models），C 盘空间有限的话建议修改。进入系统环境变量设置，为当前用户新建变量 OLLAMA_MODELS，值设为你想要存储模型的路径即可

### 1.3下载并运行模型
打开终端，运行：
```bash
ollama run qwen2.5
```
这会自动下载 Qwen2.5 模型（7B 版本约需 4GB 存储空间），下载完成后直接进入对话界面.
也可以从 Ollama 官网模型库 https://ollama.com/library 查看所有可用模型，常用命令包括：

    ollama run deepseek-r1:14b 拉取并运行 DeepSeek-R1 14B 模型

    ollama pull <model> 仅下载模型不运行

    ollama list 查看已下载的模型
**从huggingface下载模型**：
方式一：直接用 ollama run 从 Hugging Face 拉取（推荐）
这是最便捷的方式，Ollama 会直接从 Hugging Face 下载并加载模型，无需手动干预。
命令格式：
```bash

ollama run hf.co/{用户名}/{仓库名}:{量化标签}
```
实际操作示例：
以 Hugging Face 上 Unsloth 发布的 Qwen3.6-35B-A3B-GGUF 模型为例，直接运行：
```bash
ollama run hf.co/unsloth/Qwen3.6-35B-A3B-GGUF:UD-Q4_K_XL
```
更简单的获取命令的方法：打开 Hugging Face 模型页面，点击 "Use this model" 下拉框，在列表中选择 Ollama，然后选择你想要的量化版本，页面会自动生成对应的 ollama run 命令，复制执行即可，这样可以避免仓库名或量化标签拼写出错。
方式二：手动下载 GGUF 文件 + Modelfile 导入
如果你已经手动下载了 GGUF 文件，或者需要更精细地控制模型参数，可以用这种方式。
步骤 1：下载 GGUF 文件
从 Hugging Face 模型页面的 "Files and versions" 中下载 .gguf 格式的文件，例如 model-Q4_K_M.gguf。
步骤 2：创建 Modelfile
在 GGUF 文件所在目录下，创建一个名为 Modelfile 的文本文件，内容只需一行，指向 GGUF 文件的路径：
```text

FROM ./model-Q4_K_M.gguf
```
如果需要设置对话模板或采样参数，也可以额外添加 TEMPLATE、PARAMETER 等指令。
步骤 3：导入并运行
```bash
# 导入模型（my-model 是你自定义的模型名称）
ollama create my-model -f Modelfile

# 运行
ollama run my-model
```
> 注意以上方式必须是GGUF 格式的模型文件

**通用方案：使用 llama.cpp 手动转换**
如果 Ollama 无法直接导入，那么使用 llama.cpp 的官方转换脚本是标准且可靠的做法。

核心步骤：

(1)获取 llama.cpp：从官方仓库克隆源码，并安装必要的 Python 依赖。
```bash
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
pip install -r requirements.txt
```
这里的依赖主要包括 torch、transformers、safetensors 等。

(2)下载原始模型：使用 huggingface-cli 工具将 Safetensors 模型下载到本地。
```bash
huggingface-cli download meta-llama/Llama-3.1-8B --local-dir ./my-model-safetensors
```
(3)执行转换：运行 convert_hf_to_gguf.py 脚本，将 Safetensors 转换为 FP16 或 BF16 格式的 GGUF 文件。
```bash
python convert_hf_to_gguf.py ./my-model-safetensors/ --outfile ./my-model-f16.gguf --outtype f16
```
(4)（可选）量化：转换出的 FP16 GGUF 文件体积较大，可以使用 llama-quantize 工具将其压缩到更小的量化格式，如 Q4_K_M。
```bash

./llama-quantize ./my-model-f16.gguf ./my-model-Q4_K_M.gguf Q4_K_M
```
(5)导入 Ollama：最后，像上一节介绍的那样，创建一个 Modelfile 指向生成的 GGUF 文件，然后用 ollama create 导入。
```text

# Modelfile 内容
FROM ./my-model-Q4_K_M.gguf
```
### 1.4通过API调用
Ollama 默认在 http://localhost:11434 提供 OpenAI 兼容 API，可以直接用 curl 或 Python 调用：
```bash
curl http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5",
    "messages": [{"role": "user", "content": "你好"}]
  }'
```
如果你想让局域网内其他设备也能访问，需要设置环境变量 OLLAMA_HOST=0.0.0.0:11434 后重启 Ollama 服务

## 2.LM Studio——图形界面操作
LM Studio 提供完整的图形界面，支持模型下载、加载、参数调整和本地 API 服务，适合不喜欢命令行的用户。在 AMD 硬件平台上兼容性表现良好，运行稳定.
### 2.1 下载安装
访问 `https://lmstudio.ai/`，选择对应操作系统的版本下载并安装。
### 2.2 基本配置

打开 LM Studio 后，点击左下角配置按钮进入设置界面：

- 语言切换：可选简体中文

- 开启开发者模式：打开 Developer Mode，解锁 Runtime 等高级配置

- 运行时引擎选择：GGUF 格式模型默认使用 llama.cpp 引擎，推荐选择 Vulkan 后端

- 硬件调度：在 Hardware 页面开启 GPU 加速，可实时查看硬件占用状态

### 2.3 下载与加载模型

LM Studio 内置了模型下载功能，会自动根据你的硬件配置推荐适配的模型版本。你也可以从 Hugging Face 手动下载 GGUF 格式模型文件，放入软件配置的指定模型存储目录即可识别。

加载模型时，点击 “Load Model” 选择已下载的模型，调整上下文长度、GPU 层数等参数后即可开始对话。
### 2.4 启动本地 API 服务

在 LM Studio 中启动本地服务器，默认监听端口可自定义。它会提供 OpenAI 兼容的 API 端点，局域网内其他设备也可以调用


## 3.llama.cpp——资源受限环境的首选
llama.cpp 是一个用 C++ 实现的高效推理引擎，支持纯 CPU 推理和 GPU 加速（CUDA、Metal、Vulkan），适合没有高端显卡或需要完全离线运行的场景。Q4 量化后 7B 模型仅需约 6GB 内存即可运行。

### 3.1安装方式

**Windows 用户**：可以通过 WinGet 安装：
```bash

winget install llama.cpp
```
**macOS/Linux 用户**：使用一键安装脚本：
```bash

curl -LsSf https://llama.app/install.sh | sh
```
也可以从源码编译。Windows 环境下使用 CMake 构建：
```bash

git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
mkdir build && cd build
cmake .. -G "Visual Studio 18 2026" -A x64 -DLLAMA_CURL=OFF
cmake --build . --config Release
```
如果需要 GPU 加速（NVIDIA 显卡），在 CMake 配置中添加 -DLLAMA_CUDA=ON

### 3.2下载 GGUF 模型
llama.cpp 使用 GGUF 格式的模型文件。可以从 Hugging Face 或 ModelScope 下载：
```bash

# 从 Hugging Face 下载（以 DeepSeek-R1-Distill-Qwen-1.5B 为例）
wget https://huggingface.co/unsloth/DeepSeek-R1-Distill-Qwen-1.5B-GGUF/resolve/main/DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf

# 或从 ModelScope 下载（国内速度更快）
pip install modelscope
modelscope download --model Xorbits/Qwen-7B-Chat-GGUF
```
选择量化版本时，Q4_K_M 是推荐格式，在质量和体积之间取得了良好平衡
### 3.3 运行模型
直接命令行对话：
```bash

llama-cli -m qwen.gguf -i -c 4096
```
启动 API 服务：
```bash

# CPU 模式
llama-server -m qwen.gguf --host 127.0.0.1 --port 8080 -c 4096

# GPU 加速模式（将所有层 offload 到 GPU）
llama-server -m qwen-7b-chat.Q4_0.gguf -c 4096 --n-gpu-layers -1
```
服务启动后默认监听 http://localhost:8080，提供 OpenAI 兼容 API。

使用 Docker 部署（生产环境推荐） ：可以使用 Docker Compose 固定版本镜像，实现安全加固的容器化部署，适合企业知识库、智能客服等场景。

## 4.vLLM——生产级高并发部署
vLLM 是面向生产环境的高性能推理框架，并发吞吐量远超其他方案（实测 vLLM 可达 793 TPS，而 Ollama 约为 41 TPS）。它主要面向 Linux 环境，支持 NVIDIA GPU 和 AMD GPU。
### 4.1 环境要求
- 操作系统：Linux（Windows 可通过 WSL2 使用）

- Python：3.10 - 3.13

- GPU：NVIDIA（CUDA）或 AMD（ROCm）
### 4.2 安装 vLLM
```bash
# 创建虚拟环境并安装 vLLM
conda create -n vllm_env python=3.10 -y
#查看本机cuda版本
nvidia-smi
#根据输出中的 CUDA Version 选择对应的 PyTorch index
pip install vllm --extra-index-url https://download.pytorch.org/whl/cu126
```

### 4.3 启动推理服务
```bash
# 启动 vLLM 推理服务（以 DeepSeek-R1-Distill-Qwen-7B 为例）
vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 1
```
服务默认监听 http://localhost:8000，提供 OpenAI 兼容 API。多卡场景下通过 --tensor-parallel-size N 指定 GPU 数量

### 4.4 内网离线部署
如果目标服务器无法访问外网，可以在联网机器上先拉取 Docker 镜像和模型文件，再传输到内网服务器：
```bash
# 联网机器上拉取镜像
docker pull vllm/vllm-openai:latest
docker save vllm/vllm-openai:latest -o vllm.tar

# 下载模型文件
pip install modelscope
modelscope download --model Qwen/Qwen3-32B

# 将镜像和模型传输到内网后导入
docker load -i vllm.tar
```

这种方法适合企业内网、金融、政务等对数据安全要求高的场景

---

**NONONO,出现了问题，以上均只是下载使用本地配置大模型的一些整合程度比较高的方法和应用，平常写python代码调用有其他的方式，接下来容我一一介绍**
## 1.transformers + AutoModel（最标准，适合开发和实验）
这是 Hugging Face 生态的标准做法。适合单条推理、模型调试、微调等场景。
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_path = "./SearchR1-model"   # 本地模型目录

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,   # 根据硬件选 float16 / bfloat16
    device_map="auto",            # 自动分配到 GPU/CPU
    trust_remote_code=True,
)

# 构造对话
messages = [{"role": "user", "content": "你好，请介绍一下自己"}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=256, temperature=0.7)
response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
print(response)
```

关键点：

- `AutoTokenizer` 负责文本 ↔ token 的转换，`AutoModelForCausalLM` 负责推理。

- `apply_chat_template` 会按模型自带的对话模板格式化输入，这比手动拼 prompt 可靠得多。

- `trust_remote_code=True` 在加载带自定义代码的模型（如部分 Qwen、DeepSeek 变体）时经常需要。
**变体**：也可以用 pipeline 进一步简化：
```python
from transformers import pipeline

pipe = pipeline("text-generation", model=model_path, device_map="auto")
result = pipe("你好", max_new_tokens=128)
print(result[0]["generated_text"])
```
**缺点**：没有并发优化，批量推理效率低，显存管理较粗糙。适合开发阶段，不适合生产服务。

## 2.vLLM 的离线 LLM 类（高吞吐批量推理）
vLLM 除了 vllm serve 启动 API 服务，也提供 Python 离线推理接口，底层自动做 PagedAttention、连续批处理，吞吐量远高于 transformers。
```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="./SearchR1-model",
    tensor_parallel_size=1,
    gpu_memory_utilization=0.9,
    max_model_len=8192,
    trust_remote_code=True,
)

sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=512)

# 支持批量输入，vLLM 会自动调度
prompts = ["请解释什么是量子计算", "写一首关于秋天的诗"]
outputs = llm.generate(prompts, sampling_params)

for out in outputs:
    print(out.outputs[0].text)
```

**关键点：**

- 构造参数与 vllm serve 的引擎参数完全一致。

- llm.generate() 接受列表，内部自动批处理，适合一次处理成百上千条 prompt。

- 也支持 llm.chat() 直接传对话格式。

- 加载 GGUF 或 AWQ 量化模型时，传 quantization="awq" 或路径带 :Q4_K_M 即可。

**适用：离线批量数据处理、评测、需要高吞吐的 Python 服务。**

## 3.调用本地 API 服务（Ollama / vLLM / LM Studio）
如果本地已经用 Ollama、vLLM 或 LM Studio 起了服务，Python 代码里就不需要加载模型，直接用 HTTP 客户端调用即可
```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:11434/v1",   # Ollama 默认地址
    api_key="ollama",                        # Ollama 不校验，随便填
)

response = client.chat.completions.create(
    model="qwen2.5",
    messages=[{"role": "user", "content": "你好"}],
    temperature=0.7,
)
print(response.choices[0].message.content)
```

## 4.SentenceTransformers（嵌入模型专用）
如果本地模型是嵌入模型（embedding model），比如 `BAAI/bge-large-zh`、`moka-ai/m3e-base`，用 `sentence-transformers` 最方便。
```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("./bge-large-zh")

sentences = ["今天天气很好", "明天会下雨吗"]
embeddings = model.encode(sentences, normalize_embeddings=True)

print(embeddings.shape)   # (2, 1024)
```
**关键点：**

- 专为语义嵌入设计，输出的是向量，不是生成文本。

- 常用于 RAG 的检索阶段，配合向量数据库使用。

- 也支持 model.similarity() 直接算相似度。

**适用：RAG 检索、语义搜索、文本聚类、去重。**


