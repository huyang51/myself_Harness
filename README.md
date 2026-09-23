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