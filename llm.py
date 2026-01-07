from langchain_openai import ChatOpenAI
import os

ds_api_key = os.getenv("DEEPSEEK_API_KEY")

def get_llm():
    return ChatOpenAI(
        model="deepseek-chat",
        api_key=ds_api_key,
        base_url="https://api.deepseek.com",
        temperature=0
    )
