import requests
import os

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

response = requests.post(
    "https://api.groq.com/openai/v1/chat/completions",
    headers={
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    },
    json={
        "model": "llama3-8b-8192",
        "messages": [
            {"role": "system", "content": "You are a financial analyst."},
            {"role": "user", "content": "What is a fixed income bond?"}
        ]
    }
)

print(response.status_code)
print(response.json())
