import streamlit as st

def render_message(message):
  with st.chat_message(message['role']):
    st.write(message['content'])
    
    if message['id'] != st.session_state.greeting_id and message['role'] == 'assistant':
      st.session_state.feedbacks[message['id']] = st.feedback('thumbs', key=message['id'])
      if message['id'] == st.session_state.messages[-1]['id'] and 'referenced_docs' in st.session_state:
        with st.popover('References'):
          st.write(st.session_state.referenced_docs[0])
          for doc in st.session_state.referenced_docs[1:]:
            st.divider()
            st.write(doc)

def render_conversation():
  # Display chat messages from history
  for message in st.session_state.messages:
    render_message(message)