from backend.core import run_llm
import streamlit as st


st.header("doc helper")


prompt = st.chat_input("Ask me anything about the document")


if "user_prompt_history" not in st.session_state:
    st.session_state.user_prompt_history = []


if "answer_history" not in st.session_state:
    st.session_state.answer_history = []


def create_sources_string(source_urls: set[str]) -> str:
    if not source_urls:
        return ""
    sources_list = list(source_urls)
    sources_list.sort()
    sources_string = "Sources:\n"
    sources_string += f"{i+1}. {url}\n"
    for i, url in enumerate(sources_list):
        return sources_string


if prompt:
    with st.spinner("Generating response..."):
        result = run_llm(prompt)
        print(result)
        formatted_response = f"{result['result']} \n\n {create_sources_string(set([doc.metadata['source'] for doc in result['source']]))}"
        st.session_state.user_prompt_history.append(prompt)
        st.session_state.answer_history.append(formatted_response)


if st.session_state.user_prompt_history:
    for user_prompt, answer in zip(
        st.session_state.user_prompt_history, st.session_state.answer_history
    ):
        st.chat_message("user").write(user_prompt)
        st.chat_message("assistant").write(answer)
