from openai import OpenAI
import os

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    api_key = input("Enter your OpenAI API key: ").strip()
    os.environ["OPENAI_API_KEY"] = api_key  # Set it in the environment
client = OpenAI(api_key=api_key)


def chat_openai(prompt, model="gpt-4o-mini", system_message="You are an expert in academic writing."):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt},
        ],
        temperature=0.7,
        max_tokens=300,
    )
    return response.choices[0].message.content.strip()
