from typing import Dict, List
from dotenv import load_dotenv
import os

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever

load_dotenv()


def run_llm(query: str, chat_history: List[Dict[str, any]] = []):
    embeddings = GoogleGenerativeAIEmbeddings(
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        model="models/text-embedding-004",
    )

    doc_search = PineconeVectorStore(
        index_name=os.getenv("PINECONE_INDEX_NAME"), embedding=embeddings
    )

    chat = ChatGoogleGenerativeAI(
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        model="gemini-2.0-flash",
        temperature=0,
        verbose=True,
    )

    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    stuff_documents_chain = create_stuff_documents_chain(
        llm=chat,
        prompt=retrieval_qa_chat_prompt,
    )

    rephrase_prompt = hub.pull("langchain-ai/chat-langchain-rephrase")
    history_aware_retriever = create_history_aware_retriever(
        llm=chat,
        retriever=doc_search.as_retriever(),
        prompt=rephrase_prompt,
    )

    qa = create_retrieval_chain(
        retriever=history_aware_retriever,
        combine_docs_chain=stuff_documents_chain,
    )

    result = qa.invoke({"input": query, "chat_history": chat_history})

    new_result = {
        "query": result["input"],
        "result": result["answer"],
        "source": result["context"],
    }

    return new_result


if __name__ == "__main__":
    result = run_llm("What is langchain?")
    print(result)
