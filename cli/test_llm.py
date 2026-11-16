import os

from dotenv import load_dotenv
from openai import OpenAI
from openai.types.chat import ChatCompletion, ChatCompletionMessageParam


class LLMConnector:
    def __init__(self):
        load_dotenv()
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY environment variable not set")

        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )

    def send_message(self, user_prompt: str) -> ChatCompletion:
        messages: list[ChatCompletionMessageParam] = [
                {
                    "role": "system",
                    "content": "",
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
        ]

        response: ChatCompletion = self.client.chat.completions.create(
                model = "openrouter/free",
                messages = messages,
        )
        return response

def main() -> None:
    llm_connector = LLMConnector()
    response = llm_connector.send_message("Why is Boot.dev such a great place to learn about RAG? Use one paragraph maximum.")
    print(response.choices[0].message.content)
    
    if response.usage:
        print(f"Prompt tokens: {response.usage.prompt_tokens}")
        print(f"Response tokens: {response.usage.completion_tokens}")

if __name__ == "__main__":
    main()
