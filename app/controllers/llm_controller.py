from openai import OpenAI

# Set your OpenAI API key here or load it from an environment variable
openai_api_key = "***REMOVED***"

client = OpenAI(api_key=openai_api_key)

class LLMController:
    def get_response(self, text: str) -> str:
        """
        Sends the given text to GPT-4 and returns the generated response.
        """
        try:
            gpt_assistant_prompt = "Answer user questions."
            gpt_user_prompt = text
            messages = [
                {"role": "assistant", "content": gpt_assistant_prompt},
                {"role": "user", "content": gpt_user_prompt}
            ]

            # Call the OpenAI API to generate a response
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0.2,
                max_tokens=100,
                frequency_penalty=0.0
            )
            response_text = response.choices[0].message.content
            tokens_used = response.usage.total_tokens
            
            return response_text, tokens_used

        except Exception as e:
            return f"Error: {e}"
