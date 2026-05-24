"""RAG chain building module.

Constructs the LangChain Retrieval-Augmented Generation chain:
retriever → prompt → LLM → output parser.
"""

from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama


def _format_docs(docs: list) -> str:
    """Format retrieved documents into a single context string.

    Args:
        docs: List of LangChain Document objects retrieved by the retriever.

    Returns:
        A single string with document contents separated by markers.
    """
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


def build_rag_chain(
    vectorstore: Chroma,
    llm_model: str,
    base_url: str,
) -> RunnablePassthrough:
    """Build the RAG chain: retriever → prompt → LLM → output parser.

    The chain retrieves relevant document chunks from ChromaDB,
    injects them into the system prompt as context, and generates
    a grounded answer using the local LLM.

    Args:
        vectorstore: ChromaDB vector store containing document embeddings.
        llm_model: Name of the Ollama chat model (e.g. llama3).
        base_url: Ollama server URL.

    Returns:
        A LangChain Runnable chain ready for .invoke(question) calls.
    """
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4},
    )

    llm = ChatOllama(
        model=llm_model,
        base_url=base_url,
        temperature=0.1,
    )

    system_prompt = (
        "You are a helpful knowledge base assistant. "
        "Answer the user's question based ONLY on the provided context "
        "from the PDF document. "
        "If the context does not contain enough information to answer "
        "the question, say 'I cannot find the answer in the provided "
        "document.' "
        "Do NOT make up information or use outside knowledge.\n\n"
        "Context:\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{question}"),
    ])

    rag_chain = (
        {
            "context": retriever | _format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain