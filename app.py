from dotenv import load_dotenv
import os

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL_NAME = "gemini-1.5-flash-8b"
EMBEDDING_MODEL = "models/embedding-001"

import streamlit as st
import pandas as pd
from langchain.text_splitter import CharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.callbacks.base import BaseCallbackHandler


from vertexai.preview import tokenization

# uses https://github.com/google/sentencepiece under the hood, works offline
tokenizer = tokenization.get_tokenizer_for_model("gemini-1.5-flash-002")


class CustomHandler(BaseCallbackHandler):
    def on_llm_start(self, serialized, prompts, **kwargs):
        formatted_prompts = "\n".join(prompts)
        st.session_state.tokens_used = tokenizer.count_tokens(
            formatted_prompts
        ).total_tokens

    def on_chain_start(self, serialized, inputs, **kwargs):
        st.session_state.retrieved = inputs["context"]
        # st.session_state.docs = docs


handler = CustomHandler()


def get_vector_store(text_chunks):
    # print(text_chunks)
    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL, google_api_key=GEMINI_API_KEY
    )
    print(embeddings)
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)

    return vectorstore


def get_conversation_chain(vector_store):
    llm = ChatGoogleGenerativeAI(
        google_api_key=GEMINI_API_KEY,
        model=GEMINI_MODEL_NAME,
        temperature=0,
        callbacks=[handler],
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        callbacks=[handler],
    )
    system_template = """
        Use the following pieces of context and chat history to answer the question at the end.
        In the answer tell exact paragraph number from the context where the answer is found e.g. "According to §3(2) of the AGH Study Regulations, students must submit their course selection within two weeks of the semester start."
        If you don't know the answer, just say that you don't know, don't try to make up an answer.

        Context: {context}

        Chat history: {chat_history}

        Question: {question}
        
        Helpful Answer:
    """

    prompt = PromptTemplate(
        template=system_template,
        input_variables=["context", "question", "chat_history"],
        callbacks=[handler],
    )

    conversation_chain = ConversationalRetrievalChain.from_llm(
        verbose=True,
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt},
        callbacks=[handler],
    )

    return conversation_chain


def handle_user_input(question):
    response = st.session_state.conversation({"question": question})
    st.session_state.chat_history = response["chat_history"]


def display_chat_history():
    if st.session_state.chat_history:
        for msg in st.session_state.chat_history:
            if msg.type == "human":
                with st.chat_message("user"):
                    st.markdown(msg.content)
            elif msg.type == "ai":
                with st.chat_message("assistant"):
                    st.markdown(msg.content)


def main():
    st.set_page_config(page_title="AGH Regulations Assistant", page_icon=":books:")
    st.header("AGH Regulations Assistant :books:")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    if "tokens_used" not in st.session_state:  # Initialize tokens_used in session state
        st.session_state.tokens_used = 0
    if "retrieved" not in st.session_state:  # Initialize tokens_used in session state
        st.session_state.retrieved = []
    if "vector_store" not in st.session_state:
        try:
            df = pd.read_csv("data/chunks.csv")
            text_chunks = df["chunk"].tolist()
            st.session_state.vector_store = get_vector_store(text_chunks)
            st.success(
                "Your PDFs have been processed successfully. You can ask questions now."
            )
        except Exception as e:
            st.error(f"An error occurred while processing the PDF: {e}")
            return

    if st.session_state.conversation is None and st.session_state.vector_store:
        st.session_state.conversation = get_conversation_chain(
            st.session_state.vector_store
        )

    display_chat_history()

    if prompt := st.chat_input("Ask anything to your PDF:"):
        with st.chat_message("user"):
            st.markdown(prompt)
        handle_user_input(prompt)
        if st.session_state.chat_history:
            last_ai_response = [
                msg for msg in st.session_state.chat_history if msg.type == "ai"
            ][-1].content
            with st.chat_message("assistant"):
                st.markdown(last_ai_response)

    with st.sidebar:
        st.subheader("Additional Informations")
        st.markdown(f"Token Count: {st.session_state.tokens_used}\n")
        st.markdown(f"Prompt Sent to LLM: {prompt}\n")
        st.markdown(f"Retrieval:\n {st.session_state.retrieved}")


if __name__ == "__main__":
    main()
