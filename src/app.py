import streamlit as st
from vectorstore import VectorStore
from chatbot import Chatbot
import os

def main():
    st.set_page_config(page_title="Document QA Bot 🤖", layout="wide")
    st.title("Document QA Bot 🤖")
    st.write("Upload a PDF, input your API keys, and ask questions!")

    # Sidebar configuration inputs
    st.sidebar.title("Configuration")
    cohere_api_key = st.sidebar.text_input("Cohere API Key", type="password")
    pinecone_api_key = st.sidebar.text_input("Pinecone API Key", type="password")

    # File uploader widget
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

    if uploaded_file and cohere_api_key and pinecone_api_key:
        # Save file locally to process it
        with open("uploaded_document.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Initialize vector database backend
        with st.spinner("Processing document and indexing vectors..."):
            try:
                vectorstore = VectorStore("uploaded_document.pdf", cohere_api_key, pinecone_api_key)
                st.success("Document successfully processed and indexed!")
            except Exception as e:
                st.error(f"Error initializing Vector Store: {str(e)}")
                return

        # Initialize Chatbot
        chatbot = Chatbot(vectorstore, cohere_api_key)

        # Query UI input box
        user_query = st.text_input("Ask a question based on the document")
        submit_button = st.button("Submit")

        if submit_button and user_query:
            st.write(f"**You:** {user_query}")
            
            with st.spinner("Thinking..."):
                response, retrieved_docs = chatbot.respond(user_query)
                
                # Directly write out the clean answer on the webpage UI
                st.write(f"**Bot:** {response}")

                # FIX: Changed from st.expanders to singular st.expander
                with st.expander("View Referenced Document Sections"):
                    for idx, doc in enumerate(retrieved_docs):
                        st.markdown(f"**Reference {idx+1}:** {doc['text']}")

    elif not (cohere_api_key and pinecone_api_key):
        st.info("Please enter your Cohere and Pinecone API keys in the sidebar menu to proceed.")

if __name__ == "__main__":
    main()
