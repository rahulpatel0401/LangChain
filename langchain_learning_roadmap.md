# 🦜🔗 LangChain Zero-to-Hero Roadmap (Local Models Edition)

**Goal:** Master LangChain, LangGraph, and AI Engineering using **only local models (Ollama)** and free tools.
**Duration:** 8–12 Weeks (10–15 hours/week)
**Stack:** Python, VS Code, Ollama, LangChain, LangGraph.

---

## 🛠️ Phase 0: Environment Setup (Week 1)

**Objective:** Configure a robust local AI development environment.

### 1. Install Ollama & Models
1.  Download **Ollama** from [ollama.com](https://ollama.com).
2.  Pull the necessary models (llama3.1, mistral, nomic-embed-text).

### 2. Python Environment (VS Code)
1.  Create a project folder.
2.  Create a virtual environment.
3.  Install core libraries: `langchain`, `langchain-ollama`, `langgraph`, `chromadb`, `jupyter`.

### 3. VS Code Configuration
*   Install **Python** and **Jupyter** extensions.
*   Create a `.env` file.
*   Test setup with a simple script to ensure Ollama is reachable.

---

## 📚 Module 1: LangChain Basics & Models (Week 1)

**Learning Objectives:**
*   Understand what LangChain is and why it's needed.
*   Difference between LLMs (text-in/text-out) and Chat Models (messages-in/message-out).
*   Using `ChatOllama`.

**Resources:**
*   [LangChain Concepts](https://python.langchain.com/docs/concepts/)
*   [LangChain Quickstart](https://python.langchain.com/docs/tutorials/llm_chain/)

**Hands-on Exercise:**
Create a script that compares responses from two different local models (e.g., Llama 3.1 vs Mistral) for the same query.

**Mini-Project:**
**"Local CLI Chatbot"**: Build a simple Python script that runs in a loop, taking user input and printing the model's response until the user types "exit".

---

## 📝 Module 2: Prompts & Output Parsers (Week 2)

**Learning Objectives:**
*   Master `PromptTemplate` and `ChatPromptTemplate`.
*   Learn how to structure inputs (System vs. Human messages).
*   Use `StrOutputParser` and structured parsers (JSON).

**Resources:**
*   [Prompts Guide](https://python.langchain.com/docs/concepts/#prompt-templates)
*   [Output Parsers](https://python.langchain.com/docs/concepts/#output-parsers)

**Hands-on Exercise:**
Create a prompt that acts as a code reviewer, taking a code snippet as input and providing feedback.

**Mini-Project:**
**"Tweet Generator"**: A script that takes a topic and a tone (e.g., "funny", "professional") and generates 3 variations of a tweet using a structured prompt.

---

## 🔗 Module 3: Chains & LCEL (Week 3)

**Learning Objectives:**
*   Understand **LCEL** (LangChain Expression Language) - the core of modern LangChain.
*   Pipe operator `|`.
*   `RunnablePassthrough`, `RunnableLambda`.

**Resources:**
*   [LCEL Cheatsheet](https://python.langchain.com/docs/how_to/lcel_cheatsheet/)
*   [Why LCEL?](https://python.langchain.com/docs/concepts/#langchain-expression-language-lcel)

**Hands-on Exercise:**
Create a multi-step chain: Translate text -> Summarize it.

**Mini-Project:**
**"Language Tutor"**: A chain that takes a sentence in English, translates it to French, explains the grammar, and then translates it back to English to verify accuracy.

---

## 🧠 Module 4: Memory (Week 4)

**Learning Objectives:**
*   Understand why LLMs are stateless.
*   Implement `RunnableWithMessageHistory` (the modern way).
*   Use `ChatMessageHistory` to store session data.

**Resources:**
*   [Memory in LCEL](https://python.langchain.com/docs/how_to/message_history/)

**Hands-on Exercise:**
Create a chatbot that remembers your name and previous turns in the conversation.

**Mini-Project:**
**"Context-Aware Chatbot"**: A CLI bot that maintains conversation history across multiple turns.

---

## 📚 Module 5: Retrieval Augmented Generation (RAG) (Week 5-6)

**Learning Objectives:**
*   Document Loading (`TextLoader`, `PyPDFLoader`).
*   Text Splitting (`RecursiveCharacterTextSplitter`).
*   Embeddings (`OllamaEmbeddings` with `nomic-embed-text`).
*   Vector Stores (`Chroma` or `FAISS`).
*   Building a RAG Chain.

**Resources:**
*   [RAG Tutorial](https://python.langchain.com/docs/tutorials/rag/)

**Hands-on Exercise:**
Load a text file (e.g., a speech or article), ingest it into a vector store, and ask questions about it.

**Mini-Project:**
**"DocuChat"**: A script where you provide a path to a folder of text files, and it lets you chat with them.

---

## 🤖 Module 6: Agents & Tools (Week 7)

**Learning Objectives:**
*   Concept of Agents (LLM as reasoning engine).
*   Defining Tools (`@tool`).
*   Binding tools to models.
*   Using `create_tool_calling_agent`.

**Resources:**
*   [Tool Calling](https://python.langchain.com/docs/how_to/tool_calling/)

**Hands-on Exercise:**
Create a simple math agent using custom tools for basic arithmetic operations.

**Mini-Project:**
**"Math Wizard"**: An agent that can solve complex word problems by breaking them down into tool calls (Add, Multiply, Power, etc.).

---

## 🕸️ Module 7: LangGraph (Week 8-9)

**Learning Objectives:**
*   Move from `AgentExecutor` (legacy) to `LangGraph` (modern).
*   `StateGraph`, `Nodes`, `Edges`.
*   Building a ReAct Agent from scratch.

**Resources:**
*   [LangGraph Tutorial](https://langchain-ai.github.io/langgraph/tutorials/introduction/)

**Hands-on Exercise:**
Build a simple agent loop with LangGraph that takes a message and returns a response.

**Mini-Project:**
**"Agentic Researcher"**: An agent that can search the web (simulated tool) and summarize findings using a loop.

---

## 📊 Module 8: Evaluation & LangSmith (Week 10)

**Learning Objectives:**
*   Setting up LangSmith (Free Tier).
*   Tracing chains and agents.
*   Creating datasets and running evaluations.

**Resources:**
*   [LangSmith Walkthrough](https://docs.smith.langchain.com/)

**Hands-on Exercise:**
Sign up for LangSmith, configure tracing, and view the traces of your previous chains.

**Mini-Project:**
**"Eval Suite"**: Create a dataset of 10 questions for your RAG app and run an evaluation to check answer quality.

---

## 🚀 Module 9: Deployment with LangServe (Week 11)

**Learning Objectives:**
*   Turning chains into REST APIs.
*   Using `LangServe` with FastAPI.
*   Using the Playground.

**Resources:**
*   [LangServe Docs](https://python.langchain.com/docs/langserve/)

**Hands-on Exercise:**
Deploy a simple chain as a REST API using LangServe and test it via the auto-generated playground.

**Mini-Project:**
**"API-fied Chatbot"**: Deploy your RAG chatbot as an API and call it from a separate script (or Postman).

---

## 🔮 Module 10: Advanced Topics (Week 12)

**Learning Objectives:**
*   **Streaming**: `chain.stream()` for real-time output.
*   **Multi-Agent Systems**: Two agents talking to each other (e.g., Researcher + Writer).
*   **Human-in-the-loop**: Pausing LangGraph for user approval.

**Hands-on Exercise:**
Implement streaming for a long generation task.

---

## 🏆 Capstone Projects

Choose one (or all) to solidify your skills.

### 1. Personal AI Assistant 🤖
*   **Features**: Chat interface, Memory, Calculator tool, Web Search tool (simulated or using DuckDuckGo), System info tool.
*   **Tech**: LangGraph, Ollama, Streamlit (optional for UI).

### 2. "Chat with Your Life" RAG App 📄
*   **Features**: Ingests your PDF resumes, notes, and text files. Answers questions with citations (source document names).
*   **Tech**: Chroma, RecursiveCharacterTextSplitter, RetrievalChain.

### 3. Multi-Agent Research Team 🕵️‍♂️✍️
*   **Features**:
    *   **Agent A (Researcher)**: Searches for info on a topic.
    *   **Agent B (Writer)**: Takes info and writes a blog post.
    *   **Supervisor**: Critiques the blog post and asks for revisions.
*   **Tech**: LangGraph (StateGraph), Conditional Edges.
