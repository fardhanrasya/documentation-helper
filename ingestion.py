from dotenv import load_dotenv
import os

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import ReadTheDocsLoader
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    model="models/text-embedding-004",
)


def ingest_docs():
    loader = ReadTheDocsLoader(
        "langchain-docs/api.python.langchain.com/en/latest", encoding="utf-8"
    )

    raw_documents = loader.load()
    print(f"loaded {len(raw_documents)} documents")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=50)
    documents = text_splitter.split_documents(raw_documents)

    for doc in documents:
        new_url = doc.metadata["source"].replace("langchain-docs", "https:/")
        doc.metadata.update({"source": new_url})

    print(f"going to add {len(documents)} documents to the vector database")

    # Membagi dokumen menjadi batch yang lebih kecil
    batch_size = 100
    for i in range(0, len(documents), batch_size):
        batch = documents[i : i + batch_size]
        print(
            f"Processing batch {i//batch_size + 1} of {(len(documents) + batch_size - 1)//batch_size}"
        )
        PineconeVectorStore.from_documents(
            batch,
            embeddings,
            index_name=os.getenv("PINECONE_INDEX_NAME"),
        )

    print("vectorstore done.")


if __name__ == "__main__":
    ingest_docs()
