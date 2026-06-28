from groq import Groq
from decouple import config

client = Groq(
    api_key=config("GROQ_API_KEY"),
)

chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": "Explain the importance of low latency LLMs",
        }
    ],
    model=config("GROQ_MODEL")
)

print(chat_completion.choices[0].message.content)