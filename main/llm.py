from utils import Singleton
from langchain_openai import AzureChatOpenAI
from config import configuration

class Llm(metaclass=Singleton):
    def __init__(self):
        self.model = AzureChatOpenAI(
            azure_endpoint=configuration['open_ai_cred']['OPENAI_API_BASE'],
            azure_deployment=configuration['open_ai_config']['deployment_name'],
            api_key=configuration['open_ai_cred']['OPENAI_API_KEY'],
            api_version=configuration['open_ai_config']['openai_api_version'])
        
        self.model_mini = AzureChatOpenAI(
            azure_endpoint=configuration['open_ai_cred']['OPENAI_API_BASE'],
            azure_deployment=configuration['open_ai_config']['deployment_name_mini'],
            api_key=configuration['open_ai_cred']['OPENAI_API_KEY'],
            api_version=configuration['open_ai_config']['openai_api_version'])