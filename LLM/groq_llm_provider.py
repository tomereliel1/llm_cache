import os

from groq import APIConnectionError, APIStatusError, Groq

from .i_llm_provider import ILLMProvider


class GroqLLMProvider(ILLMProvider):
    def __init__(self, model_name="llama-3.1-8b-instant"):
        self.model_name = model_name
        self.api_key = os.environ.get("GROQ_API_KEY")

        if not self.api_key:
            raise RuntimeError("GROQ_API_KEY is not set. Run: export GROQ_API_KEY='your_api_key'")

        self.client = Groq(api_key=self.api_key)

    def generate_answer(self, prompt):
        print("Prompt received. Sending request to Groq, this may take a while...")

        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                temperature=0.5,
                max_completion_tokens=1024,
                top_p=1,
            )
        except APIStatusError as error:
            raise RuntimeError(
                f"Groq API request failed with status {error.status_code}: {error.response.text}"
            ) from error
        except APIConnectionError as error:
            raise RuntimeError(f"Could not connect to Groq API: {error}") from error

        return completion.choices[0].message.content
