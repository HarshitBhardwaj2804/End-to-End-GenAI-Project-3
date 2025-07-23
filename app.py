## Creating the Search Engine GenAI App.
import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler
import os
from dotenv import load_dotenv
load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

## Langsmith tracking
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Search Engine GenAI App"

arxiv_api_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
arxiv_tool = ArxivQueryRun(api_wrapper=arxiv_api_wrapper)

wikipedia_api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
wikipedia_tool = WikipediaQueryRun(api_wrapper = wikipedia_api_wrapper)

search = DuckDuckGoSearchRun(name = "Search")

## Streamlit interface

st.title("Search Engine GenAI App")

st.sidebar.title("Settings")
model = st.sidebar.selectbox("Model", ["Meta llama3", "Google Gemma2","Meta llama4"])

llm_dict = {
    "Meta llama3": "llama3-8b-8192",
    "Google Gemma2": "Gemma2-9b-It",
    "Meta llama4": "meta-llama/llama-4-scout-17b-16e-instruct"
}

model_name = llm_dict[model]

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role":"assistant", "content":"Hi, I'm a chatbot who can search the web. How can I assist you today?"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt:= st.chat_input(placeholder="What is machine learning?"):
    st.session_state.messages.append({"role":"user", "content":prompt})
    st.chat_message("user").write(prompt)

    model = ChatGroq(groq_api_key=groq_api_key, model=model_name, streaming=True)
    tools = [arxiv_tool, wikipedia_tool, search]

    search_agent = initialize_agent(tools, model, agent = AgentType.ZERO_SHOT_REACT_DESCRIPTION, handling_parsing = True)

    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        response = search_agent.run(st.session_state.messages, callbacks=[st_cb])
        st.session_state.messages.append({"role":"assistant", "content":response})
        st.write(response)






