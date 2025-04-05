import streamlit as st
import utils.i18n as i18n
from utils.helper import add_user_message, add_assistant_message, get_response
from .conversation import render_message

def render_chat_input():
  # Get and display user's question with the answer
  prompt_content = st.chat_input(i18n.prompt_hint)
  if prompt_content:
    # Add user input to chat history
    user_prompt = add_user_message(prompt_content)
    render_message(user_prompt)
    response = add_assistant_message(get_response(user_prompt['content']))
    render_message(response) # should be write_stream, change when update the get_response to model StreamError