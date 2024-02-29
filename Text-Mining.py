#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains.react import ReAct
from langchain.embeddings import SBERTEmbeddings
from langchain.llms import GPT4All
from langchain.util import langchain_to_streamlit

# Global variable for memory
memory = []

def main():
    # Streamlit page configuration
    st.set_page_config(page_title="CHATBOT for legal queries related to divorce and inheritance", page_icon=':books:')
    st.header("CHATBOT for legal queries related to divorce and inheritance :books:")

    # Upload PDF documents
    pdf = st.file_uploader("Upload your documents", type="pdf", accept_multiple_files=True)

    # Extract text from uploaded PDFs
    texts = []
    if pdf is not None:
        for pdf_file in pdf:
            pdf_reader = PdfReader(pdf_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            texts.append(text)

    # Split text into chunks
    chunks = []
    for text in texts:
        text_splitter = CharacterTextSplitter(separator="\n\n", chunk_size=1000, chunk_overlap=200, length_function=len)
        chunks.extend(text_splitter.split_text(text))

    # Create embeddings using SBERT
    embeddings = SBERTEmbeddings(model_name="paraphrase-multilingual-MiniLM-L12-v2")

    # Create context using Chroma from texts
    context = Chroma.from_texts(chunks, embeddings)

    # Ask user for input query
    query = st.text_input("Ask a question about your documents:")

    # Process query and provide response
    if query:
        # Get relevant documents from context
        docs = context.similarity_search(query)

        # Initialize GPT4All language model
        gpt = GPT4All()

        # Initialize ReAct framework
        react = ReAct(gpt)

        # Generate response
        response = react.answer_query(docs, query)

        # Store the question and answer in memory
        memory.append({"question": query, "answer": response})

        # Display response
        st.write(response)

    # Memory section
    if st.checkbox("Memory"):
        st.write("Memory Selected!")
        # Display the memory
        if memory:
            st.write("Previous Questions and Answers:")
            for i, entry in enumerate(memory, 1):
                st.write(f"Q{i}: {entry['question']}")
                st.write(f"A{i}: {entry['answer']}")
        else:
            st.write("No previous questions and answers in memory.")

    # Reset memory
    if st.button("Clear Memory"):
        memory.clear()



if __name__ == '__main__':
    main()


# In[ ]:




