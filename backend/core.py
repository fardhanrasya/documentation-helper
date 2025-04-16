from dotenv import load_dotenv
import os

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain

load_dotenv()


def run_llm(query: str):
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

    qa = create_retrieval_chain(
        retriever=doc_search.as_retriever(),
        combine_docs_chain=stuff_documents_chain,
    )

    result = qa.invoke({"input": query})

    new_result = {
        "query": result["input"],
        "result": result["answer"],
        "source": result["context"],
    }

    return new_result


if __name__ == "__main__":
    result = run_llm("What is langchain?")
    print(result)
