import streamlit as st
from utils.helper import add_user_message, add_assistant_message, get_response
  
def click_question(question):
  add_user_message(question)
  response = get_response(question)
  add_assistant_message(response)

def render_followup_questions():
  if 'followup_questions' in st.session_state:
    for question in st.session_state.followup_questions:
      st.button(
        question,
        on_click=click_question,
        args=[question],
        use_container_width=True
      )