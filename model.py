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
    openai_api_key = 'sk-367e1eb0bd604aedb010909f9e6df83d'
    openai_api_base = 'https://api.deepseek.com'
    model_name = 'deepseek-flash'
    client = OpenAI(api_key=openai_api_key,
                    base_url=openai_api_base)

    try:
        response = send_test_request(client=client,model_name=model_name)
        logger.info(response)
    except Exception as e:
        logger.error(e)