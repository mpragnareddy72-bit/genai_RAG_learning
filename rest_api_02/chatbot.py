import requests

URL = "http://localhost:11434/api/chat"

messages = [
    {
        "role": "system",
        "content": "You are a kind and helpful assistant."
    }
]

while True:

    user_message = input("You: ")

    if user_message.lower() == "exit":
        break

    messages.append({
        "role": "user",
        "content": user_message
    })

    payload = {
        "model": "llama3.2",
        "messages": messages,
        "stream": False
    }

    response = requests.post(
        URL,
        json=payload
    )

    response.raise_for_status()

    data = response.json()

    assistant_message = data["message"]["content"]

    print("Assistant:", assistant_message)

    messages.append({
        "role": "assistant",
        "content": assistant_message
    })

