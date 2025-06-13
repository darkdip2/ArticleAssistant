import os
import google.generativeai as genai
from google.generativeai import types
from dotenv import load_dotenv


LLMCALLS_LIMIT = 9
MODELS = ['gemini-2.0-flash', 'gemini-2.0-flash', 'gemini-1.5-flash']

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env file")

genai.configure(api_key=GEMINI_API_KEY)

def llm(contents: list, temperature: float = 0.53, tokens: int = 2048) -> str:
    if not hasattr(llm, "call_count"):
        llm.call_count = 0

    generation_config = types.GenerationConfig(
        temperature=temperature,
        max_output_tokens=tokens
    )

    system_prompt = (
        "You are a Retrieval-Augmented Generation (RAG) chatbot for Occam's Advisory. "
        "Your role is to help users with accurate answers based on the provided context. "
        "If the answer is unknown or outside the given context, clearly respond that you do not know. "
        "Do not hallucinate or guess."
    )

    full_contents = [{"role": "assistant", "parts": [{"text": system_prompt}]}] + contents

    for model_name in MODELS:
        for _ in range(3):
            if llm.call_count >= LLMCALLS_LIMIT:
                return 'LLM call failed'

            try:
                model = genai.GenerativeModel(model_name)
                response = model.generate_content(
                    full_contents,
                    generation_config=generation_config
                )
                llm.call_count += 1

                if response.candidates and response.candidates[0].content.parts:
                    return response.candidates[0].content.parts[0].text

            except Exception as e:
                print(e)
                return 'LLM call failed'