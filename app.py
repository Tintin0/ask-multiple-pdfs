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
    if llm_choice == "gpt-4":
        llm = ChatOpenAI(model_name="gpt-4")
    elif llm_choice == "gpt-3.5-turbo":
        llm = ChatOpenAI(model_name="gpt-3.5-turbo")
    elif llm_choice == "flan-t5-xxl":
        llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})
    elif llm_choice == "MistralAI":
        llm = HuggingFaceHub(repo_id="mistralai/Mistral-7B-v0.1", model_kwargs={"temperature":0.5, "max_length":512})

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
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)

def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs and Word DOCX",
                       page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("FLIX Legal Team: Chat with multiple PDFs and Word DOCX:books:")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")

        # LLM-Auswahl
        llm_choices = ["gpt-4", "gpt-3.5-turbo", "flan-t5-xxl", "MistralAI"]
        selected_llm = st.selectbox("Wählen Sie LLM:", llm_choices)

        if "selected_llm" not in st.session_state:
            st.session_state.selected_llm = selected_llm
        else:
            if st.session_state.selected_llm != selected_llm:
                st.session_state.conversation = None
                st.session_state.chat_history = None
                st.session_state.selected_llm = selected_llm

        pdf_docs = st.file_uploader(
            "Laden Sie hier Ihre PDFs oder DOCX-Dateien hoch und klicken Sie auf 'analysieren'",
            accept_multiple_files=True,
            type=['pdf', 'docx']
        )

        if st.button("Analysieren"):
            with st.spinner("Analysiere"):
                # Holen Sie sich den Dokumenttext
                raw_text = get_document_text(pdf_docs)

                # Holen Sie sich die Textblöcke
                text_chunks = get_text_chunks(raw_text)

                # Vektor-Store erstellen
                vectorstore = get_vectorstore(text_chunks)

                # Konversationskette erstellen
                st.session_state.conversation = get_conversation_chain(
                    vectorstore, st.session_state.selected_llm)

if __name__ == '__main__':
    main()





# import streamlit as st
# from dotenv import load_dotenv
# from pypdf import PdfReader
# from langchain.text_splitter import CharacterTextSplitter
# from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
# from langchain.vectorstores import FAISS
# from langchain.chat_models import ChatOpenAI
# from langchain.memory import ConversationBufferMemory
# from langchain.chains import ConversationalRetrievalChain
# from htmlTemplates import css, bot_template, user_template
# from docx import Document
#
# from langchain.llms import HuggingFaceHub
#
# # def get_pdf_text(pdf_docs):
# #     text = ""
# #     for pdf in pdf_docs:
# #         pdf_reader = PdfReader(pdf)
# #         for page in pdf_reader.pages:
# #             text += page.extract_text()
# #     return text
#
# def get_document_text(docs):
#     text = ""
#     for doc in docs:
#         if doc.type == "application/pdf":
#             pdf_reader = PdfReader(doc)
#             for page in pdf_reader.pages:
#                 text += page.extract_text()
#         elif doc.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":  # DOCX
#             docx_doc = Document(doc)
#             for paragraph in docx_doc.paragraphs:
#                 text += paragraph.text + "\n"
#     return text
#
# def get_text_chunks(text):
#     text_splitter = CharacterTextSplitter(
#         separator="\n",
#         chunk_size=1000,
#         chunk_overlap=200,
#         length_function=len
#     )
#     chunks = text_splitter.split_text(text)
#     return chunks
#
#
# def get_vectorstore(text_chunks):
#     embeddings = OpenAIEmbeddings()
#     # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
#     vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
#     return vectorstore
#
#
# def get_conversation_chain(vectorstore):
#     #llm = ChatOpenAI(model_name="gpt-3.5-turbo")
#     llm = ChatOpenAI(model_name="gpt-4")
#     # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})
#
#     memory = ConversationBufferMemory(
#         memory_key='chat_history', return_messages=True)
#     conversation_chain = ConversationalRetrievalChain.from_llm(
#         llm=llm,
#         retriever=vectorstore.as_retriever(),
#         memory=memory
#     )
#     return conversation_chain
#
#
# def handle_userinput(user_question):
#     response = st.session_state.conversation({'question': user_question})
#     st.session_state.chat_history = response['chat_history']
#
#     for i, message in enumerate(st.session_state.chat_history):
#         if i % 2 == 0:
#             st.write(user_template.replace(
#                 "{{MSG}}", message.content), unsafe_allow_html=True)
#         else:
#             st.write(bot_template.replace(
#                 "{{MSG}}", message.content), unsafe_allow_html=True)
#
#
# def main():
#     load_dotenv()
#     st.set_page_config(page_title="Chat with multiple PDFs and Word DOCX",
#                        page_icon=":books:")
#     st.write(css, unsafe_allow_html=True)
#
#     if "conversation" not in st.session_state:
#         st.session_state.conversation = None
#     if "chat_history" not in st.session_state:
#         st.session_state.chat_history = None
#
#     st.header("Chat with multiple PDFs and Word DOCX:books:")
#     user_question = st.text_input("Ask a question about your documents:")
#     if user_question:
#         handle_userinput(user_question)
#
#     with st.sidebar:
#         st.subheader("Your documents")
#         pdf_docs = st.file_uploader(
#             "Upload your PDFs or DOCX files here and click on 'analyze'",
#             accept_multiple_files=True,
#             type=['pdf', 'docx']
#         )
#
#         if st.button("Analyze"):
#             with st.spinner("Analyzing"):
#                 # get pdf text
#                 raw_text = get_document_text(pdf_docs)
#
#                 # get the text chunks
#                 text_chunks = get_text_chunks(raw_text)
#
#                 # create vector store
#                 vectorstore = get_vectorstore(text_chunks)
#
#                 # create conversation chain
#                 st.session_state.conversation = get_conversation_chain(
#                     vectorstore)
#
#
# if __name__ == '__main__':
#     main()
