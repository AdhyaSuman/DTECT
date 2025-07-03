from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.language_models.chat_models import BaseChatModel
from typing import List


class ChatGemini(BaseChatModel):
    def __init__(self, api_key: str, model: str = "gemini-pro", temperature: float = 0.7):
        self.model = model
        self.temperature = temperature
        self.api_key = api_key
        self.client = ChatGoogleGenerativeAI(
            model=model,
            temperature=temperature,
            google_api_key=api_key
        )

    def _generate(self, messages: List, stop: List[str] = None):
        # Convert LangChain messages to string
        prompt = "\n".join(
            msg.content for msg in messages if isinstance(msg, (HumanMessage, AIMessage))
        )
        response = self.client.invoke(prompt)
        return response

    @property
    def _llm_type(self) -> str:
        return "gemini"
