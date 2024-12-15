import streamlit as st
from mistralai import Mistral
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler
from langchain.llms import MistralAI
from langchain.tools import DuckDuckGoSearchRun

mistral_api_key = os.getenv('MISTRAL_API_KEY')
if not mistral_api_key:
    mistral_api_key = st.secrets.get("MISTRAL_API_KEY", None)
    
with st.sidebar:
    #mistral_api_key = st.text_input(
    #    "Mistral AI API Key", key="langchain_search_api_key_mistral", type="password"
    #)
    "[Get a Mistral AI API key](https://console.mistral.ai/)"
    "[View the source code](https://github.com/streamlit/llm-examples/blob/main/pages/2_Chat_with_search.py)"
    "[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/streamlit/llm-examples?quickstart=1)"

st.title("🔎 LangChain - Chat with search (Mistral AI)")

"""
In this example, we're using `StreamlitCallbackHandler` to display the thoughts and actions of an agent in an interactive Streamlit app.
This version uses Mistral AI instead of OpenAI.
"""

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi, I'm a chatbot powered by Mistral AI who can search the web. How can I help you?"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input(placeholder="Who won the Women's U.S. Open in 2018?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    if not mistral_api_key:
        st.info("Please add your Mistral AI API key to continue.")
        st.stop()

    # Initialize Mistral client
    mistral_client = Mistral(api_key=mistral_api_key)

    # Create a MistralAI LLM instance for LangChain
    llm = MistralAI(mistral_client=mistral_client, model="mistral-large-latest")

    search = DuckDuckGoSearchRun(name="Search")
    search_agent = initialize_agent(
        [search], llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, handle_parsing_errors=True
    )
    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        response = search_agent.run(st.session_state.messages, callbacks=[st_cb])
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.write(response)
