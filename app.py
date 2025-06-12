from dotenv import load_dotenv
import os

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL_NAME = "gemini-1.5-flash-8b"
EMBEDDING_MODEL = "models/embedding-001"
MIN_KEYWORDS_SCORE = 0.15
MIN_UNCERTAINTY_SCORE = 0.8
MIN_CONTEXT_OVERLAP = 0.5

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

from evaluation import ResponseEvaluator

# uses https://github.com/google/sentencepiece under the hood, works offline
tokenizer = tokenization.get_tokenizer_for_model("gemini-1.5-flash-002")




class CustomHandler(BaseCallbackHandler):
    def on_llm_start(self, serialized, prompts, **kwargs):
        formatted_prompts = "\n".join(prompts)
        st.session_state.tokens_used = tokenizer.count_tokens(
            formatted_prompts
        ).total_tokens
        st.session_state.add_info_prompt = prompts[0]

    def on_chain_start(self, serialized, inputs, **kwargs):
        st.session_state.retrieved = inputs["context"]
        # st.session_state.docs = docs


handler = CustomHandler()
evaluator = ResponseEvaluator(
    min_length=20,
    max_length=50,
)


def get_vector_store(text_chunks):
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
        Use the following pieces of context and chat history to answer in English the question at the end.
        In the answer tell exact paragraph number from the context where the answer is found e.g. "According to §3(2) of the AGH Study Regulations, students must submit their course selection within two weeks of the semester start."
        If the answer have paragraph number (e.g. "§3(2)"), you should include in anserw this document name "AGH Study Regulations".
        If the answer starts with FACT AND FIGURES or ACADEMIC YEAR SCHEDULE then add those name instead of paragraph number.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        Answer in full sentences.

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
    ai_msgs = [msg for msg in st.session_state.chat_history if msg.type == "ai"]
    last_ai_response = ai_msgs[-1].content if ai_msgs else ""
    return last_ai_response


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
    if "add_info_prompt" not in st.session_state:
        st.session_state.add_info_prompt = None
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
        last_ai_response = handle_user_input(prompt)
        if last_ai_response:
            with st.chat_message("assistant"):
                st.markdown(last_ai_response)

        if isinstance(st.session_state.retrieved, list):
            context_text = " ".join(st.session_state.retrieved)
        else:
            context_text = str(st.session_state.retrieved)

        eval_res = evaluator.evaluate(prompt, last_ai_response, context_text)

        warning_msgs = []
        if not eval_res['basic_criteria']:
            warning_msgs.append(eval_res['message'])
        else:
            keyword_score = eval_res['keyword_score']
            length_score = eval_res['length_score']
            uncertainty_score = eval_res['uncertainty_score']

            if keyword_score < MIN_KEYWORDS_SCORE:
                warning_msgs.append(
                    f"Low keyword match score: {keyword_score}. "
                )
            if length_score == -1:
                warning_msgs.append(
                    f"Answer is too short."
                )
            if length_score == 1:
                warning_msgs.append(
                    f"Answer is too long."
                )
            if uncertainty_score < MIN_UNCERTAINTY_SCORE:
                warning_msgs.append(
                    f"High uncertainty in the answer: {uncertainty_score}. "
                )
            if eval_res['context_overlap_score'] < MIN_CONTEXT_OVERLAP:
                warning_msgs.append(
                    f"Low context overlap score: {eval_res['context_overlap_score']}. "
                )
        for wm in warning_msgs:
            st.warning(wm)

    with st.sidebar:
        st.header("Additional Information")

        with st.expander("💸 Token Count"):
            st.code(st.session_state.tokens_used, language="markdown")

        with st.expander("💬 Prompt Sent to LLM"):
            st.code(st.session_state.add_info_prompt, language="markdown")

        with st.expander("📄 Retrieved Documents"):
            st.markdown("Chunks retrieved by the vector search:")
            st.code(st.session_state.retrieved, language="markdown")



if __name__ == "__main__":
    main()
