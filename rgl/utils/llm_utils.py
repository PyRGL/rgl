from openai import OpenAI
import os





def get_openai_api_key():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        api_key = input("Enter your OpenAI API key: ").strip()
    return api_key

def get_openai_client(api_key=None):
    
    if api_key is None:
        api_key = get_openai_api_key()

    client = OpenAI(api_key=api_key)
    return client


def chat_openai(prompt, 
                model="gpt-4o-mini", 
                sys_prompt="",
                client=None):


    if client is None:
        client = get_openai_client()        
    
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": prompt},
        ],
        temperature=0.7,
        max_tokens=300,
    )
    return response.choices[0].message.content.strip()
