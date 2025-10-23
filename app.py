

import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv
import os


load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


st.set_page_config(page_title="💬 Gemini Conversational Chatbot")
st.title("💬 Conversational Q&A Chatbot (Google Gemini)")


st.sidebar.header("⚙️ Model Settings")

model_name = st.sidebar.selectbox(
    "Select a Gemini model",
    ["gemini-1.5-flash", "gemini-1.5-pro"]
)

temperature = st.sidebar.slider(
    "Temperature (creativity)",
    0.0, 1.0, 0.7,
    help="Higher = more creative, Lower = more focused"
)


if "model" not in st.session_state:
    st.session_state.model = genai.GenerativeModel(model_name)

if "chat" not in st.session_state:
    st.session_state.chat = st.session_state.model.start_chat(history=[])

if "history" not in st.session_state:
    st.session_state.history = []  


user_input = st.chat_input("Type your message...")

if user_input:
   
    st.chat_message("user").write(user_input)
    st.session_state.history.append(("user", user_input))

   
    response = st.session_state.chat.send_message(user_input)
    ai_message = response.text

    
    st.chat_message("assistant").write(ai_message)
    st.session_state.history.append(("assistant", ai_message))

if st.session_state.history:
    st.write("### 🕒 Conversation History")
    for role, message in st.session_state.history:
        if role == "user":
            st.markdown(f"**🧑 You:** {message}")
        else:
            st.markdown(f"**🤖 Gemini:** {message}")
