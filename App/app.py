import os
import streamlit as st
from dotenv import load_dotenv, find_dotenv
from urllib.parse import urlparse, parse_qs
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_core.tools import tool
from langchain.tools import Tool
from duckduckgo_search import DDGS
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage
from langchain.memory import ConversationBufferWindowMemory
from langchain_openai import ChatOpenAI
import requests
from PIL import Image
from io import BytesIO

_ = load_dotenv(find_dotenv())

OPENAI_API_KEY  = os.getenv('OPENAI_API_KEY') 
LANGCHAIN_API_KEY = os.getenv('LANGCHAIN_API_KEY')
os.environ["LANGCHAIN_TRACING_V2"] = "true"
embedding_model = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

faiss_index_path = r"C:\Users\tiago\OneDrive\Ambiente de Trabalho\Last Project\parent-helper\faiss_youtube_index"
#vectorstore = FAISS.from_documents(split_docs, embedding_model)
vectorstore = FAISS.load_local(faiss_index_path, embeddings=embedding_model, allow_dangerous_deserialization=True)


# chat completion llm
llm = ChatOpenAI(
    openai_api_key=OPENAI_API_KEY,
    model_name='gpt-4o-mini',  # Ensure this model is supported
    temperature=0.0
)

# conversational memory
memory = ConversationBufferWindowMemory(
    memory_key='chat_history',
    k=5,
    return_messages=True
)

from langchain.chains import RetrievalQA
# Create a retriever from the loaded FAISS vector store
faiss_retriever = vectorstore.as_retriever(
    search_type="similarity", 
    search_kwargs={'k': 5}     # Number of documents to retrieve
)

# Initialize the QA chain with the FAISS retriever
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff", 
    retriever=faiss_retriever,
    return_source_documents=True # Optional: To see which chunks were retrieved
)


@tool
def calculator(expression: str) -> str:
    """Safely evaluate a basic school-level math expression like '2 + 3 * 4'."""
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"Error: {e}"

@tool
def document_search(query: str) -> str:
    """Searches the local vector store for relevant documents based on the query."""
    # Use the existing RetrievalQA chain
    response = qa_chain.invoke(query)
    return response

@tool
def image_search(query: str) -> str:
    """Search for a safe, educational image to help explain the topic visually."""
    try:
        with DDGS() as ddgs:
            results = ddgs.images(query, max_results=3, safesearch="moderate")
            if results:
                for item in results:
                    image_url = item['image']  
                    response = requests.head(image_url, allow_redirects=True, timeout=5)
                    if response.status_code == 200:
                        return f"![Visual Aid]({image_url})"
    except Exception as e:
        return f"Image search failed: {str(e)}"

    return "No relevant image found."

tools = [calculator, document_search, image_search]


tools_str = ", ".join([tool.name for tool in tools])

system_prompt = f'''
You are a warm, friendly, and funny assistant designed to help parents teach educational topics to their children.

You have access to the following tools: {tools_str}

**Key Objectives**:
- You **must** use the **Document Search** tool first.
- Immediately after, you **must** call the **Image Search** tool to find an image,a diagram that supports the explanation.
- These two tools are always used together, in that order, unless the topic is purely mathematical.

- Use the **Calculator** only for math-related questions.
- Use language that is **clear, fun, and child-friendly** so parents can easily explain topics to kids aged 6–10.

**If Document Search returns "No result found"**, respond with:
"I couldn't find any direct information about {{input}}, but maybe this helps:" — then give your best general answer.

**Important**:
- Use tools before giving your final answer.
- Use kid-friendly language and metaphors like:  
  - “Imagine your body is like a machine...”  
  - “It’s kind of like when you...”  
  - “Let’s pretend...”
- Always include a visual (image or fallback text).
- Your final answer should be short and written in regular adult language.

**Language Behavior**:
Always respond in the same language as the user.

---

Use the following format for every interaction:

Question: {{input}}
Thought: Think about which tool to use first (always start with Document Search).
Action: Select one of [{tools_str}]
Action Input: Write the query you are sending to the tool based on the input
Observation: The result returned by the tool

...(repeat Thought/Action/Observation as needed — **Document Search first, then Image Search**)...

Thought: I now know the final answer

Answer starts with:

[Category_name]

Let me check the educational materials...

**Explanation:**
- [Explain using info from Document Search, in kids' language and with bullet points]
- [Use examples, simple words, metaphors]
- [Explain math steps if relevant]

Let me find an image to help explain this visually...

**Visual Aid:**
![Visual Aid](image_url_here)  
(or write: "No relevant image or diagram was found.")

**Answer:**
[A short, direct summary in adult language]

---

Begin!

Question: {{input}}
Thought: {{agent_scratchpad}}
'''

custom_prompt = ChatPromptTemplate.from_template(system_prompt) #alternative to ChatPromptTemplate.from_messages that uses the place holders better 

agent = create_openai_functions_agent(llm=llm, tools=tools, prompt=custom_prompt) 
agent_executor = AgentExecutor(agent=agent, tools=tools, memory=memory, verbose=True)

# App logo
st.image("logo.png", width=700)

# Store chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input and response handling
if prompt := st.chat_input("Be the one to teach your kid"):
    # Show user message
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Run the agent (tool-powered LLM logic)
    response = agent_executor.invoke({"input": prompt})

    # Show AI response
    with st.chat_message("assistant"):
        st.markdown(response["output"])
    st.session_state.messages.append({"role": "assistant", "content": response["output"]})
