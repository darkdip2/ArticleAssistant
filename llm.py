import os
import google.generativeai as genai
from google.generativeai import types
from dotenv import load_dotenv


LLMCALLS_LIMIT = 9
MODELS = ['gemini-2.0-flash', 'gemini-2.0-flash', 'gemini-1.5-flash']

load_dotenv()

GEMINI_API_KEY = None 
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:GEMINI_API_KEY='AIzaSyBkqapafNApFb85__rrueoSW-oRRRxKPmA'

genai.configure(api_key=GEMINI_API_KEY)


K = 5
def split_document(content: str) -> str:
    lines = content.split("\n")
    quarter_length = len(lines) // 4
    return "\n".join(lines[:quarter_length])

def parse_top_K_articles(docs: list) -> str:
    combined_content = "\n\n".join(split_document(doc.page_content) for doc in docs[:K])
    return combined_content

def is_followup_query(user_input: str, history: list) -> bool:
    if not history or len(history) < 2:
        return False
    prompt = f"Given the conversation history and the latest user input, determine if the latest input is a follow-up question related to the previous context.\n\nHistory:\n"
    for msg in history[-3:]:
        prompt += f"{msg['role'].capitalize()}: {msg['content']}\n"
    prompt += f"\nLatest input: {user_input}\n\nIs the latest input a follow-up question? Respond with 'Yes' or 'No'."
    response = llm([{"role": "user", "parts": [{"text": prompt}]}])
    return response.strip().lower() == "yes"

def build_llm_contents(user_input: str, reranked_text: str, history: list) -> list:
    context = reranked_text
    history_parts = []
    for msg in history[-3:]:
        if msg["role"] == "user":
            history_parts.append({"role": "user", "parts": [{"text": msg["content"]}]})
        elif msg["role"] == "assistant":
            history_parts.append({"role": "assistant", "parts": [{"text": msg["content"]}]})
    history_parts.append({"role": "user", "parts": [{"text": f"Relevant context:\n{context}"}]})
    history_parts.append({"role": "user", "parts": [{"text": user_input}]})
    return history_parts


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