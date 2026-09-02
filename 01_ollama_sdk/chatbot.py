import ollama

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

    response = ollama.chat(
        model="llama3.2",
        messages=messages
    )

    assistant_message = response["message"]["content"]

    print("Assistant:", assistant_message)

    messages.append({
        "role": "assistant",
        "content": assistant_message
    })