from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict

def create_rag_chain(vector_store):
    """
    Create a RAG chain for question answering
    
    Args:
        vector_store: Vector store containing the indexed documents
        
    Returns:
        StateGraph: Compiled RAG chain
    """
    llm = ChatOllama(
        model="deepseek-r1:1.5b",
        temperature=0.1,
    )
    
    # Modified prompt template to handle cases with no relevant information
    template = """Answer the question based only on the following context:

    Context: {context}

    Question: {question}

    Instructions:
    1. If the context doesn't contain information that helps answer the question, respond EXACTLY with "NO RELEVANT INFORMATION FOUND IN THE URLS".
    2. Don't attempt to answer if the necessary information isn't in the context.
    3. Only use information from the provided context.

    Your response:"""

    custom_rag_prompt = PromptTemplate.from_template(template)

    # Define state for application
    class State(TypedDict):
        question: str
        context: List[Document]
        answer: str

    # Define application steps
    def retrieve(state: State):
        retrieved_docs = vector_store.similarity_search(state["question"])
        return {"context": retrieved_docs}

    def generate(state: State):
        docs_content = "\n\n".join(doc.page_content for doc in state["context"])
        messages = custom_rag_prompt.invoke({"question": state["question"], "context": docs_content})
        response = llm.invoke(messages)
        return {"answer": response.content}

    # Compile application
    graph_builder = StateGraph(State).add_sequence([retrieve, generate])
    graph_builder.add_edge(START, "retrieve")
    graph = graph_builder.compile()
    
    return graph