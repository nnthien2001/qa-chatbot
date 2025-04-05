import streamlit as st
from .IdGenerator import IdGenerator
from . import i18n
from model import model

def init():
  if 'id_generator' not in st.session_state:
    config_path = 'source/configs/gcp2.env'
    config = model.load_config(config_path)
    buffer = model.init(config)
    st.session_state.config = config
    st.session_state.buffer = buffer
    
    st.session_state.id_generator = IdGenerator()
    st.session_state.messages = []
    message = add_assistant_message(i18n.greeting)
    st.session_state.greeting_id = message['id']
    
    # a dictionary with key is the message id and value is the feedback value
    st.session_state.feedbacks = {}

def add_message(role: str, content: str):
  if 'id_generator' not in st.session_state:
    raise Exception('id_generator does not exist in session state. Please check the initialisation.')
  
  message = {
    'id': st.session_state.id_generator.next(),
    'role': role,
    'content': content
  }
  st.session_state.messages.append(message)
  return message

def add_assistant_message(content: str):
  return add_message('assistant', content)

def add_user_message(content: str):
  return add_message('user', content)

def get_response(prompt):
  response, buffer = model.chat(st.session_state.config, prompt, st.session_state.buffer)
  st.session_state.followup_questions = response['followup_questions']
  st.session_state.referenced_docs = [doc.get_text() for doc in response['additional_nodes']]
  st.session_state.buffer = buffer
  return response['response']