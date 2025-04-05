import streamlit as st

def get_response(user_input):
  chat_history = st.session_state.messages
  retrieved_context = ''
  
  template = f'''
You are a helpful assistant. Based on the following chat history and additional context, 
please provide a comprehensive response to the user's question.\n\n"
Chat History:\n{chat_history}\n\n
Additional Context:\n{retrieved_context}\n\n
User's Question: {user_input}\n
Answer:
'''

  return 'Sorry, I don\'t understand that.'

def get_follow_up_questions(prompt, response):
  follow_ups = [
    'Hi, how are you?',
    'Skibidi',
    'It\'s a plane, it\'s a bird, it\'s Superman!',
    'Spider-Man, Spider-Man, Does whatever a spider can.'
  ]
  
  return follow_ups