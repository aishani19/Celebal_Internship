import streamlit as st
from src.api_utils import get_api_response



def display_chat_interface():
    
    st.markdown("<h2 style='color: red;'>Chat Assistant</h2>", unsafe_allow_html=True)
    st.markdown("<h4 style='color: orange;'>Type a Query</h4>", unsafe_allow_html=True)


    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "session_id" not in st.session_state:
        st.session_state.session_id = None
    if "model" not in st.session_state:
        st.session_state.model = "llama-3.3-70b-versatile"

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

   
    prompt = None

    # === Text input ===
    if text_input := st.chat_input("Type your query here"):
        prompt = text_input

    # === Process prompt and generate response ===
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.spinner("Generating response..."):
            response = get_api_response(prompt, st.session_state.session_id, st.session_state.model)

        if response:
            st.session_state.session_id = response.get('session_id')
            st.session_state.messages.append({"role": "assistant", "content": response['answer']})

            with st.chat_message("assistant"):
                st.markdown(response['answer'])


        else:
            st.error("Failed to get a response from the API.")
