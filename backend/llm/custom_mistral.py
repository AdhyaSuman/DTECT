from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.outputs import ChatResult, ChatGeneration
import requests
import os

class ChatMistral(BaseChatModel):
    def __init__(self, hf_token=None, model_url=None):
        self.hf_token = hf_token or os.getenv("HF_TOKEN")
        self.model_url = model_url or "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.1"
        self.headers = {"Authorization": f"Bearer {self.hf_token}"}

    def _call(self, prompt: str) -> str:
        response = requests.post(
            self.model_url,
            headers=self.headers,
            json={"inputs": prompt, "parameters": {"max_new_tokens": 256}},
        )
        return response.json()[0]["generated_text"]

    def invoke(self, messages, **kwargs):
        prompt = "\n".join([msg.content for msg in messages if isinstance(msg, HumanMessage)])
        response = self._call(prompt)
        return AIMessage(content=response)

    def _generate(self, messages, stop=None, **kwargs) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=self.invoke(messages))])
