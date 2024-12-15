import os
from mistralai import Mistral
import streamlit as st

with st.sidebar:
    mistral_api_key = st.text_input("Mistral AI API Key", key="chatbot_api_key", type="password")
    "[Get a Mistral AI API key](https://console.mistral.ai/)"
    "[View the source code](https://github.com/streamlit/llm-examples/blob/main/Chatbot.py)"
    "[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/streamlit/llm-examples?quickstart=1)"

st.title("💬 Chatbot")
st.caption("🚀 A Streamlit chatbot powered by Mistral AI")
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
    if not mistral_api_key:
        st.info("Please add your Mistral AI API key to continue.")
        st.stop()

    client = Mistral(api_key=mistral_api_key)
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    
    chat_response = client.chat.complete(
        model="mistral-large-latest",  # или другая подходящая модель Mistral
        messages=st.session_state.messages
    )
    
    msg = chat_response.choices[0].message.content
    st.session_state.messages.append({"role": "assistant", "content": msg})
    st.chat_message("assistant").write(msg)
