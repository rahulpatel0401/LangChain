# Module 9: Deployment with LangServe (Week 11)

## Learning Objectives

By the end of this module, you will be able to:
*   **Understand the "Why" of Deployment**: Move your LangChain apps from a local Python script to a robust, shareable web API.
*   **Master LangServe**: Use the standard library for deploying LangChain runnables.
*   **Integrate with FastAPI**: Build production-ready web servers that host your chains.
*   **Utilize the Playground**: Debug and demo your chains with the automatic interactive UI.
*   **Build Client Applications**: Connect to your LangServe APIs using `RemoteRunnable`.

## Prerequisites & Setup

This week, we are stepping out of the pure "script" world and into the "web server" world. We will be using **FastAPI**, the modern standard for building Python APIs, and **LangServe**, LangChain's deployment companion.

### Installation

Open your terminal and install the necessary packages:

```bash
pip install "langserve[all]" fastapi uvicorn sse_starlette
```

*   `langserve[all]`: Installs LangServe and its client/server dependencies.
*   `fastapi`: The web framework we'll use to host the application.
*   `uvicorn`: The lightning-fast ASGI server that runs FastAPI.
*   `sse_starlette`: Required for streaming responses (Server-Sent Events).

---

## Turning Chains into REST APIs

### The Problem: "It works on my machine"
So far, all our amazing AI agents and chains have lived inside a terminal on your laptop. If you wanted to show a friend, you'd have to send them your code, have them install Python, set up an environment... it's a mess.

To make your AI useful to the world (or just a frontend web app), you need to expose it as an **API** (Application Programming Interface).

### The Solution: LangServe
**LangServe** is a library that helps you wrap your LangChain runnables (Chains, Agents, Runnables) into a REST API automatically.

It handles all the messy parts for you:
*   **Input/Output Schemas**: Automatically figures out what JSON your chain expects.
*   **Streaming**: Supports real-time token streaming out of the box.
*   **Batching**: Allows processing multiple inputs at once.
*   **Playground**: Gives you a free UI to test your bot.

Think of LangServe as the "delivery truck" that takes your "AI product" from the factory (your code) to the customer (the user's browser).

---

## Using LangServe with FastAPI

Let's build our first API. We will deploy a simple "Joke Generator" chain.

### What we're about to build
We will create a file named `server.py`. Inside, we'll define a simple chain that tells jokes about a given topic. Then, we'll use `add_routes` to "mount" this chain onto a FastAPI app. Finally, we'll run the server.

### Imports explained
*   `FastAPI`: The main class for our web application.
*   `add_routes`: The magic LangServe function that adds all the API endpoints (invoke, stream, batch, etc.) for our chain.
*   `ChatOllama`: Our local LLM.
*   `PromptTemplate`: To structure our input.
*   `StrOutputParser`: To make sure the API returns clean text, not a complex Message object.

### The Code (`server.py`)

```python
#!/usr/bin/env python
from fastapi import FastAPI
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langserve import add_routes
import uvicorn

# 1. Create the Chain
model = ChatOllama(model="llama3.1")
prompt = PromptTemplate.from_template("Tell me a short, funny joke about {topic}.")
chain = prompt | model | StrOutputParser()

# 2. Create the FastAPI App
app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="A simple API server using LangChain's Runnable interfaces",
)

# 3. Add the Chain Routes
# This adds /invoke, /batch, /stream, /playground, etc.
add_routes(
    app,
    chain,
    path="/joke",
)

if __name__ == "__main__":
    # Run the server on localhost:8000
    uvicorn.run(app, host="localhost", port=8000)
```

### How to Run It
Save this code as `server.py`. In your terminal, run:

```bash
python server.py
```

You should see output indicating the server is running, usually at `http://localhost:8000`.

### Why this works — breakdown
1.  **The Chain**: We built a standard LCEL chain (`prompt | model | parser`). It expects a dictionary `{"topic": "..."}` as input.
2.  **The App**: `app = FastAPI(...)` initializes an empty web server.
3.  **`add_routes`**: This is the heavy lifter. By calling `add_routes(app, chain, path="/joke")`, LangServe automatically created several endpoints:
    *   `POST /joke/invoke`: Wait for the full response.
    *   `POST /joke/stream`: Stream the response token by token.
    *   `POST /joke/batch`: Process a list of inputs.
    *   `GET /joke/playground`: An interactive UI.

---

## Using the Playground

One of the best features of LangServe is the **Playground**. It's a web interface that lets you interact with your API without writing any client code.

### Try it out
1.  Ensure your `server.py` is still running.
2.  Open your web browser and go to: **[http://localhost:8000/joke/playground/](http://localhost:8000/joke/playground/)**

You should see a clean UI with a generic input form.
1.  You'll see a field for `topic`.
2.  Type "ice cream".
3.  Click "Start".

**Expected Output (in browser):**
You will see the joke stream in real-time!

> "Why did the ice cream truck break down? Because there was a rocky road ahead!"

### Why is this useful?
The playground is invaluable for **debugging**. It shows you exactly what the input schema looks like and lets you verify that your chain works correctly in a deployed environment, separate from your local Python script context.

---

## Client-side Interaction (RemoteRunnable)

Now that we have a server, how do we talk to it from *another* Python script? Maybe you have a Streamlit app or a CLI tool that needs to use this API.

LangChain provides a special runnable called `RemoteRunnable`. It looks and acts exactly like a normal local chain, but when you `.invoke()` it, it makes an HTTP request to your server.

### What we're about to build
A client script `client.py` that connects to our running joke server and asks for a joke about "programming".

### Imports explained
*   `RemoteRunnable`: A class that acts as a proxy for a LangServe API.

### The Code (`client.py`)

```python
from langserve import RemoteRunnable

# Connect to the specific path we defined in the server
remote_chain = RemoteRunnable("http://localhost:8000/joke/")

print("Calling remote chain...")

# We use it JUST like a normal local chain!
response = remote_chain.invoke({"topic": "programming"})

print(f"Response: {response}")
```

### Expected Output
```text
Calling remote chain...
Response: Why do programmers prefer dark mode? Because light attracts bugs.
```

### Why this works — breakdown
*   `RemoteRunnable("http://localhost:8000/joke/")`: This tells LangChain, "I don't have the model here. The model is over there at that URL."
*   `.invoke(...)`: When you call this, `RemoteRunnable` serializes your input to JSON, sends a `POST` request to `http://localhost:8000/joke/invoke`, waits for the answer, and gives it back to you.
*   **Abstraction**: Your client code doesn't need to know *how* the joke is generated. It could be a local Llama 3, a cloud GPT-4, or a complex agent. The client just sends a topic and gets a string.

---

## Hands-on Exercise: The Translator API

Let's step it up. We will deploy a chain that translates text into a target language.

**Goal**: Create a server with a `/translate` endpoint.

1.  **Create `translator_server.py`**:
    ```python
    from fastapi import FastAPI
    from langchain_ollama import ChatOllama
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langserve import add_routes
    import uvicorn

    # 1. Define the Chain
    model = ChatOllama(model="llama3.1")
    
    # Using ChatPromptTemplate for better structure
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful translator. Translate the user's text to {language}."),
        ("human", "{text}"),
    ])
    
    chain = prompt | model | StrOutputParser()

    # 2. Setup App
    app = FastAPI(title="Translator API")

    # 3. Add Routes
    add_routes(
        app,
        chain,
        path="/translate",
    )

    if __name__ == "__main__":
        uvicorn.run(app, host="localhost", port=8001) # Note: Port 8001
    ```

2.  **Run the server**:
    ```bash
    python translator_server.py
    ```

3.  **Test via cURL (Terminal)**:
    Open a new terminal window (keep the server running) and send a raw HTTP request:
    ```bash
    curl -X POST http://localhost:8001/translate/invoke \
         -H "Content-Type: application/json" \
         -d '{"input": {"language": "French", "text": "Hello, how are you?"}}'
    ```

    **Expected JSON Output**:
    ```json
    {"output": "Bonjour, comment allez-vous ?", "callback_events": [], "metadata": {"run_id": "..."}}
    ```

    *Note: The input format `{"input": {...}}` is the standard LangServe expectation for `invoke` endpoints.*

---

## Mini-Project: "API-fied Chatbot"

For this week's capstone, we will deploy a **Context-Aware Chatbot** as an API. This is tricky because APIs are usually "stateless" (they forget everything after the request finishes).

To handle memory in an API, we usually rely on the client to pass the `chat_history`, or we use a persistent database on the server side. For simplicity and robustness, we will use the **client-managed history** approach, where the client sends the full conversation history each time.

### The Server (`chat_server.py`)

```python
#!/usr/bin/env python
from typing import List, Tuple
from fastapi import FastAPI
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langserve import add_routes
import uvicorn

# 1. Setup Model
model = ChatOllama(model="llama3.1")

# 2. Setup Prompt with History
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a friendly AI assistant named Bob."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
])

chain = prompt | model | StrOutputParser()

# 3. Define Input Schema (Optional but good practice)
# LangServe infers this, but explicit typing helps.
# We expect an input dict with "input" (str) and "chat_history" (List[Message])

app = FastAPI(title="Bob the Chatbot")

add_routes(
    app,
    chain,
    path="/chat",
)

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8002)
```

### The Client (`chat_client.py`)

This client will run in your terminal, keep track of history locally, and send it to the server with every new message.

```python
from langserve import RemoteRunnable
from langchain_core.messages import HumanMessage, AIMessage

# Connect to the server
bot = RemoteRunnable("http://localhost:8002/chat/")

chat_history = []

print("--- Chat with Bob (API Version) ---")
print("Type 'exit' to quit.")

while True:
    user_input = input("\nYou: ")
    if user_input.lower() == "exit":
        break
    
    # Prepare input for the remote chain
    # We must match the variables expected by the prompt: "input" and "chat_history"
    response = bot.invoke({
        "input": user_input,
        "chat_history": chat_history
    })
    
    print(f"Bob: {response}")
    
    # Update local history
    chat_history.append(HumanMessage(content=user_input))
    chat_history.append(AIMessage(content=response))
```

### Instructions
1.  Run `python chat_server.py` in one terminal.
2.  Run `python chat_client.py` in another terminal.
3.  Chat with Bob! Ask him his name, then ask "What did I just ask you?" to verify he has memory.

---

## Quiz & Exercises

### Quiz
1.  **What is the primary purpose of LangServe?**
    a) To train new LLMs.
    b) To deploy LangChain runnables as REST APIs.
    c) To create a frontend UI for chatbots.
    d) To speed up Python code.

2.  **Which function do we use to attach a chain to a FastAPI app?**
    a) `attach_chain()`
    b) `mount_app()`
    c) `add_routes()`
    d) `deploy_runnable()`

3.  **If my server is running at `localhost:8000/mychain`, where is the playground located?**
    a) `/mychain/ui`
    b) `/mychain/playground`
    c) `/playground`
    d) `/docs`

*(Answers: 1-b, 2-c, 3-b)*

### Challenge
Take your **RAG application** from Module 5 (the one that reads text files).
1.  Wrap it in a `server.py`.
2.  Deploy it to port `8005`.
3.  Use the Playground to ask questions about your documents.
4.  **Bonus**: Create a `RemoteRunnable` client that streams the answer back (`.stream()` instead of `.invoke()`).

---

## Further Reading & Resources

*   **[LangServe Official Documentation](https://python.langchain.com/docs/langserve/)**: The definitive guide.
*   **[FastAPI Tutorial](https://fastapi.tiangolo.com/tutorial/)**: If you want to learn more about building web APIs in Python.
*   **[LangChain Templates](https://github.com/langchain-ai/langchain/tree/master/templates)**: A collection of reference architectures deployable with LangServe.

You have now unlocked the ability to **share your AI** with the world. In the next and final module, we will tackle **Advanced Topics** like streaming (in depth), multi-agent orchestration, and human-in-the-loop workflows!
