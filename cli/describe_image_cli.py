import argparse
import base64
import mimetypes
import os
from pathlib import Path

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

    def send_message(self, user_prompt) -> ChatCompletion:
        messages: list[ChatCompletionMessageParam] = [
            {
                "role": "system",
                "content": """
                        Given the included image and text query, rewrite the text query to improve search results from a movie database. Make sure to:
                        - Synthesize visual and textual information
                        - Focus on movie-specific details (actors, scenes, style, etc.)
                        - Return only the rewritten query, without any additional commentary
                    """,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ]

        response: ChatCompletion = self.client.chat.completions.create(
            model="openrouter/free",
            messages=messages,
        )
        return response


def rewrite(llm_connector: LLMConnector, image_path: Path, query: str):
    mime = mimetypes.guess_type(image_path)[0] or "image/jpeg"
    img = Path(image_path).read_bytes()

    data_url = f"data:{mime};base64,{base64.b64encode(img).decode()}"
    response = llm_connector.send_message([
        {"type": "image_url", "image_url": {"url": data_url}},
        {"type": "text", "text": query.strip()},
    ])
    content = response.choices[0].message.content
    if content is None:
        raise RuntimeError("No text in API response")

    print(f"Rewritten query: {content.strip()}")
    if response.usage is not None:
        print(f"Total tokens:    {response.usage.total_tokens}")

def main() -> None:
    parser = argparse.ArgumentParser(description="Multimodal CLI")
    parser.add_argument("--image", type=Path, required=True, help="Image path")
    parser.add_argument("--query", type=str, required=True, help="Query to rewrite based on image")

    args = parser.parse_args()
    llm_connector = LLMConnector()

    rewrite(llm_connector, args.image, args.query)


if __name__ == "__main__":
    main()
