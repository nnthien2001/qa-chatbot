import streamlit as st
import utils.i18n as i18n
from utils.helper import add_assistant_message

def render_sidebar():
  with st.sidebar:
    if st.button(i18n.new_thread, use_container_width=True):
      st.session_state.messages = []
      message = add_assistant_message(i18n.greeting)
      st.session_state.greeting_id = message['id']
      
      st.session_state.feedbacks = {}
      st.session_state.followup_questions = []

      st.rerun() # Rerun the app to reflect changes