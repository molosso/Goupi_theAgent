"""
Interface Streamlit pour l'agent.
 
Lancement:
    streamlit run streamlit_app.py
"""
 
import streamlit as st
from agent_core import run_agent
 
st.set_page_config(page_title="Mon Agent", page_icon="🤖")
st.title("Goupi L'agent")
st.caption("Calculatrice + recherche web, propulsé par Groq (Llama 3.3 70B)")
 
if "history" not in st.session_state:
    st.session_state.history = []
 
# Affiche l'historique
for entry in st.session_state.history:
    with st.chat_message(entry["role"]):
        st.markdown(entry["content"])
 
# Zone de saisie
user_input = st.chat_input("Pose une question...")
 
if user_input:
    st.session_state.history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
 
    with st.chat_message("assistant"):
        tool_log = st.empty()
        calls_made = []
 
        def on_tool_call(name, args, result):
            calls_made.append(f"🔧 **{name}**({args}) → `{result}`")
            tool_log.markdown("\n\n".join(calls_made))
 
        with st.spinner("Réflexion..."):
            response = run_agent(user_input, on_tool_call=on_tool_call)
 
        st.markdown(response)
 
    st.session_state.history.append({"role": "assistant", "content": response})
 