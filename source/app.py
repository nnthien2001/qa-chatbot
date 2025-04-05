import streamlit as st
from utils.helper import init, add_assistant_message, add_user_message
import utils.i18n as i18n
from components.sidebar import render_sidebar
from components.conversation import render_conversation
from components.chat_input import render_chat_input
from components.followup_questions import render_followup_questions

st.set_page_config(layout='wide', page_title=i18n.page_title, page_icon=i18n.page_icon)
st.title(i18n.title)

init()

render_sidebar()
render_conversation()
render_chat_input()
if len(st.session_state.messages) > 1: # If not the first message
  render_followup_questions()