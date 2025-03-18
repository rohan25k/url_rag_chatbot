import streamlit as st
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.utils import clean_webpage_content

def load_and_process_urls(urls_list):
    """
    Load and process URLs into document chunks for the RAG model
    
    Args:
        urls_list (list): List of URLs to process
        
    Returns:
        list: Processed document chunks
    """
    with st.spinner('Loading and processing URLs...'):
        # Load content from all URLs
        loader = WebBaseLoader(web_paths=urls_list)
        docs = loader.load()

        # Clean all webpage content
        for doc in docs:
            doc.page_content = clean_webpage_content(doc.page_content)
            
        # Split the cleaned documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,  # chunk size (characters)
            chunk_overlap=200,  # chunk overlap (characters)
            add_start_index=True,  # track index in original document
        )
        all_splits = text_splitter.split_documents(docs)
        
        st.success(f'Processed {len(urls_list)} URLs into {len(all_splits)} document chunks')
        
        return all_splits