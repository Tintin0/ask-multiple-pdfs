import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import HuggingFaceHub
from htmlTemplates import css, bot_template, user_template
from docx import Document
import requests
from bs4 import BeautifulSoup

def get_text_from_url(url):
    try:
        page = requests.get(url)
        soup = BeautifulSoup(page.content, "html.parser")
        return ' '.join([p.get_text() for p in soup.find_all('p')])
    except Exception as e:
        print(f"Ein Fehler ist aufgetreten: {e}")
        return ""
def get_document_text(docs):
    text = ""
    for doc in docs:
        if doc.type == "application/pdf":
            pdf_reader = PdfReader(doc)
            for page in pdf_reader.pages:
                text += page.extract_text()
        elif doc.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":  # DOCX
            docx_doc = Document(doc)
            for paragraph in docx_doc.paragraphs:
                text += paragraph.text + "\n"
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore, llm_choice):
    if llm_choice == "gpt-4-turbo":
        llm = ChatOpenAI(model_name="gpt-4-1106-preview")
    elif llm_choice == "gpt-3.5-turbo":
        llm = ChatOpenAI(model_name="gpt-3.5-turbo")
    elif llm_choice == "flan-t5-xxl":
        llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})
    elif llm_choice == "MistralAI":
        llm = HuggingFaceHub(repo_id="HuggingFaceH4/zephyr-7b-beta", model_kwargs={"temperature":0.5, "max_length":2048})

    # Fügen Sie hier bei Bedarf weitere Auswahlmöglichkeiten hinzu

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def handle_userinput(user_question):
    if st.session_state.conversation is not None:
        response = st.session_state.conversation({'question': user_question})
        st.session_state.chat_history = response['chat_history']

        for i, message in enumerate(st.session_state.chat_history):
            if i % 2 == 0:
                st.write(user_template.replace(
                    "{{MSG}}", message.content), unsafe_allow_html=True)
            else:
                st.write(bot_template.replace(
                    "{{MSG}}", message.content), unsafe_allow_html=True)
    else:
        st.error("Die Konversationsfunktion ist noch nicht initialisiert.")

#def handle_userinput(user_question):
#    response = st.session_state.conversation({'question': user_question})
#    st.session_state.chat_history = response['chat_history']
#
#    for i, message in enumerate(st.session_state.chat_history):
#        if i % 2 == 0:
#            st.write(user_template.replace(
#                "{{MSG}}", message.content), unsafe_allow_html=True)
#        else:
#            st.write(bot_template.replace(
#                "{{MSG}}", message.content), unsafe_allow_html=True)

def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs, Word DOCX, and Web Pages",
                       page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("FLIX Legal Team: Chat with multiple PDFs, Word DOCX, and Web Pages :books:")
    user_question = st.text_input("Ask a question about your documents or web pages:")

    with st.sidebar:
        st.subheader("Your documents")

        # Datei-Upload
        pdf_docs = st.file_uploader(
            "Laden Sie hier Ihre PDFs oder DOCX-Dateien hoch",
            accept_multiple_files=True,
            type=['pdf', 'docx']
        )

        # URL-Eingabefeld für Webseiten
        url = st.text_input("Geben Sie die URL einer Webseite ein:")

        # LLM-Auswahl
        llm_choices = ["gpt-4-turbo", "gpt-3.5-turbo", "flan-t5-xxl", "MistralAI"]
        selected_llm = st.selectbox("Wählen Sie LLM:", llm_choices)

        if "selected_llm" not in st.session_state:
            st.session_state.selected_llm = selected_llm
        else:
            if st.session_state.selected_llm != selected_llm:
                st.session_state.conversation = None
                st.session_state.chat_history = None
                st.session_state.selected_llm = selected_llm

        if st.button("Analysieren"):
            with st.spinner("Analysiere"):
                raw_text = ""

                # Holen Sie sich den Text aus PDFs und DOCX-Dateien
                if pdf_docs:
                    raw_text += get_document_text(pdf_docs)

                # Fügen Sie den Text der Webseite hinzu, falls vorhanden
                if url:
                    webpage_text = get_text_from_url(url)
                    raw_text += "\n" + webpage_text

                # Holen Sie sich die Textblöcke
                text_chunks = get_text_chunks(raw_text)

                # Vektor-Store erstellen
                vectorstore = get_vectorstore(text_chunks)

                # Konversationskette erstellen
                st.session_state.conversation = get_conversation_chain(
                    vectorstore, st.session_state.selected_llm)

    if user_question and st.session_state.conversation:
        handle_userinput(user_question)

if __name__ == '__main__':
    main()
