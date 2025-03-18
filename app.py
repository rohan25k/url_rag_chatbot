import streamlit as st
from src.data_processing import load_and_process_urls
from src.rag_model import create_rag_chain
from langchain_ollama import OllamaEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore

# Initialize Streamlit app
st.set_page_config(page_title="RAG Chatbot", page_icon="🤖")
st.title("RAG Chatbot")

# Initialize session state variables
if 'urls_processed' not in st.session_state:
    st.session_state.urls_processed = False
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Main application logic
if not st.session_state.urls_processed:
    st.subheader("Step 1: Enter URLs")
    # Text area for user to input URLs
    urls_input = st.text_area(
        "Enter comma-separated URLs to process:", 
        "https://aws.amazon.com/what-is/retrieval-augmented-generation/,https://www.geeksforgeeks.org/agents-artificial-intelligence/"
    )
    
    if st.button("Process URLs"):
        # Extract URLs from input and clean up spaces
        urls_list = [url.strip() for url in urls_input.split(",") if url.strip()]
        
        if urls_list:
            # Initialize embeddings and vector store
            with st.status("Initializing embedding model and vector store..."):
                embeddings = OllamaEmbeddings(model="deepseek-r1:1.5b")
                global vector_store
                vector_store = InMemoryVectorStore(embeddings)
            
            # Load and process URLs, then add documents to vector store
            with st.status("Processing URLs and adding documents..."):
                all_splits = load_and_process_urls(urls_list)
                _ = vector_store.add_documents(documents=all_splits)
            
            # Create the RAG retrieval chain
            with st.status("Creating RAG chain..."):
                st.session_state.rag_chain = create_rag_chain(vector_store)
            
            # Mark URLs as processed and store them in session state
            st.session_state.urls_processed = True
            st.session_state.processed_urls = urls_list
            st.success("URLs processed successfully!")
            st.rerun()
        else:
            st.error("Please enter at least one valid URL")
else:
    st.subheader("Step 2: Ask Questions")
    # Display processed URLs for reference
    st.write(f"📚 Processed URLs: {', '.join(st.session_state.processed_urls)}")
    
    # Button to allow re-processing different URLs
    if st.button("Process Different URLs"):
        st.session_state.urls_processed = False
        st.session_state.chat_history = []
        st.rerun()
    
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # User input field for questions
    user_query = st.chat_input("Ask a question about the processed content...")
    
    if user_query:
        # Add user question to chat history
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        
        # Display user's question in chat
        with st.chat_message("user"):
            st.write(user_query)
        
        # Generate and display assistant's response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = st.session_state.rag_chain.invoke({"question": user_query})
                st.write(result["answer"])
                
                # Provide an option to view retrieved context
                with st.expander("Show retrieved context"):
                    st.write("\n\n".join(doc.page_content for doc in result["context"]))
        
        # Add assistant's response to chat history
        st.session_state.chat_history.append({"role": "assistant", "content": result["answer"]})
