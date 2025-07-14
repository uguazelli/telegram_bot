
from langchain_openai import ChatOpenAI
import os

openai_key = os.getenv("OPENAI_API_KEY")

# Unified assistant model for AcmeTech, guided by system context
assistant_llm = ChatOpenAI(
    openai_api_key=openai_key,
    temperature=0.3,
    model="gpt-4"
)

system_prompt = (
    "You are an AI assistant for AcmeTech Solutions, a logistics technology company.\n"
    "Answer the user's question based only on the provided CONTEXT.\n"
    "If the context does not include an answer, reply with 'I don’t know' or suggest contacting support.\n"
    "Keep responses concise, friendly, and clear.\n"
)
