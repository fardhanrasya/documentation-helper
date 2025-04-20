from dotenv import load_dotenv
import os
import re
import json

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import ReadTheDocsLoader
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import FireCrawlLoader
from langchain.docstore.document import Document
from firecrawl import FirecrawlApp

load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    model="models/text-embedding-004",
)

# Text splitter dengan ukuran yang sesuai untuk Pinecone
text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=100)

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

def ingest_docs_firecrawl() -> None:
    """
    Menggunakan FireCrawlLoader dengan mode scrape dari LangChain.
    Perhatikan bahwa fungsi ini membutuhkan firecrawl-py==0.0.20 untuk berfungsi dengan benar
    sesuai dengan dokumentasi Firecrawl (https://docs.firecrawl.dev/integrations/langchain).
    """
    docs_urls = [
        "https://python.langchain.com/docs/integrations/chat/",
        "https://python.langchain.com/docs/integrations/retrievers/",
        "https://python.langchain.com/docs/integrations/tools/",
        "https://python.langchain.com/docs/integrations/document_loaders/",
        "https://python.langchain.com/docs/integrations/vectorstores/",
        "https://python.langchain.com/docs/integrations/text_embedding/",
        "https://python.langchain.com/docs/integrations/llms/",
        "https://python.langchain.com/docs/integrations/stores/",
        "https://python.langchain.com/docs/integrations/document_transformers/",
        "https://python.langchain.com/docs/integrations/llm_caching/",
        "https://python.langchain.com/docs/integrations/graphs/",
        "https://python.langchain.com/docs/integrations/memory/",
        "https://python.langchain.com/docs/integrations/callbacks/",
        "https://python.langchain.com/docs/integrations/adapters/",
    ]
    

    total_documents_processed = 0

    for url in docs_urls:
        print(f"Processing URL with scrape mode: {url}")
        try:
            # Langsung gunakan mode scrape karena ada masalah dengan mode crawl
            loader = FireCrawlLoader(
                api_key=os.getenv("FIRECRAWL_API_KEY"),
                url=url,
                mode="scrape"  # Hanya gunakan mode scrape
            )
            
            documents = loader.load()
            print(f"Loaded {len(documents)} documents from {url}")
            
            
            for doc in documents:
                # Minimal metadata untuk menghindari batasan ukuran
                minimal_metadata = {
                    "url": url
                }
                
                # Jika ada judul, simpan judul (dengan ukuran maksimal)
                if "title" in doc.metadata and doc.metadata["title"]:
                    minimal_metadata["title"] = doc.metadata["title"][:100]
                
                # Ganti metadata lama dengan yang minimal
                doc.metadata = minimal_metadata
            
            # Gunakan text_splitter untuk memecah dokumen
            split_docs = text_splitter.split_documents(documents)
            print(f"Split into {len(split_docs)} smaller chunks with RecursiveCharacterTextSplitter")
            
            if split_docs:
                # Simpan ke Pinecone
                print(f"Storing {len(split_docs)} document chunks to Pinecone")
                
                # Split batch menjadi lebih kecil (10 dokumen)
                batch_size = 10
                for i in range(0, len(split_docs), batch_size):
                    batch = split_docs[i : i + batch_size]
                    print(
                        f"Processing batch {i//batch_size + 1} of {(len(split_docs) + batch_size - 1)//batch_size}"
                    )
                    
                    try:
                        PineconeVectorStore.from_documents(
                            batch,
                            embeddings,
                            index_name=os.getenv("PINECONE_INDEX_NAME"),
                        )
                        total_documents_processed += len(batch)
                        print(f"Successfully processed batch of {len(batch)} documents")
                    except Exception as batch_error:
                        print(f"Error processing batch: {str(batch_error)}")
                        print("Trying to process documents one by one...")
                        
                        # Jika terjadi kesalahan pada batch, coba proses satu per satu
                        for doc in batch:
                            try:
                                # Pastikan metadata tidak terlalu besar sebelum upload
                                metadata_size = len(json.dumps(doc.metadata).encode('utf-8'))
                                if metadata_size > 40000:  # Sedikit di bawah batas 40960
                                    print(f"Metadata too large ({metadata_size} bytes). Creating minimal version.")
                                    doc.metadata = {"url": url}
                                
                                PineconeVectorStore.from_documents(
                                    [doc],
                                    embeddings,
                                    index_name=os.getenv("PINECONE_INDEX_NAME"),
                                )
                                total_documents_processed += 1
                                print(f"Successfully processed document with size {len(doc.page_content)} chars")
                            except Exception as doc_error:
                                print(f"Failed to process a document: {str(doc_error)}")
                                continue
                
        except Exception as e:
            print(f"Error processing {url}: {str(e)}")
            continue
    
    print(f"Total documents processed and stored: {total_documents_processed}")
    print("Vectorstore update completed.")

def ingest_docs_direct_firecrawl() -> None:
    """
    Menggunakan FirecrawlApp langsung untuk scraping URL dan menyimpan hasilnya ke Pinecone.
    Fungsi ini menggunakan API Firecrawl secara langsung tanpa melalui LangChain.
    """
    docs_urls = [
        "https://python.langchain.com/docs/integrations/chat/",
        "https://python.langchain.com/docs/integrations/retrievers/",
        "https://python.langchain.com/docs/integrations/tools/",
        "https://python.langchain.com/docs/integrations/document_loaders/",
        "https://python.langchain.com/docs/integrations/vectorstores/",
        "https://python.langchain.com/docs/integrations/text_embedding/",
        "https://python.langchain.com/docs/integrations/llms/",
        "https://python.langchain.com/docs/integrations/stores/",
        "https://python.langchain.com/docs/integrations/document_transformers/",
        "https://python.langchain.com/docs/integrations/llm_caching/",
        "https://python.langchain.com/docs/integrations/graphs/",
        "https://python.langchain.com/docs/integrations/memory/",
        "https://python.langchain.com/docs/integrations/callbacks/",
        "https://python.langchain.com/docs/integrations/adapters/",
    ]
    
    # Inisialisasi FirecrawlApp
    app = FirecrawlApp(api_key=os.getenv("FIRECRAWL_API_KEY"))
    
    total_documents_processed = 0

    for url in docs_urls:
        print(f"Processing URL with direct Firecrawl scrape: {url}")
        try:
            # Gunakan scrape_url untuk mendapatkan konten dari URL
            scrape_result = app.scrape_url(url, params={
                'pageOptions': {
                    'onlyMainContent': True
                }
            })
            
            # Ekstrak markdown dari hasil scraping
            markdown_content = scrape_result.get("markdown", "")
            
            if not markdown_content:
                print(f"No markdown content found for {url}")
                continue
                
            # Buat metadata minimal
            metadata = {
                "url": url
            }
            
            # Tambahkan judul jika tersedia
            if "metadata" in scrape_result and "title" in scrape_result["metadata"]:
                metadata["title"] = scrape_result["metadata"]["title"][:100]
            
            # Buat dokumen LangChain dari hasil scraping
            document = Document(
                page_content=markdown_content,
                metadata=metadata
            )
            
            # Gunakan text_splitter untuk memecah dokumen
            split_docs = text_splitter.split_documents([document])
            print(f"Split into {len(split_docs)} smaller chunks with RecursiveCharacterTextSplitter")
            
            if split_docs:
                # Simpan ke Pinecone
                print(f"Storing {len(split_docs)} document chunks to Pinecone")
                
                # Split batch menjadi lebih kecil (10 dokumen)
                batch_size = 10
                for i in range(0, len(split_docs), batch_size):
                    batch = split_docs[i : i + batch_size]
                    print(
                        f"Processing batch {i//batch_size + 1} of {(len(split_docs) + batch_size - 1)//batch_size}"
                    )
                    
                    try:
                        PineconeVectorStore.from_documents(
                            batch,
                            embeddings,
                            index_name=os.getenv("PINECONE_INDEX_NAME"),
                        )
                        total_documents_processed += len(batch)
                        print(f"Successfully processed batch of {len(batch)} documents")
                    except Exception as batch_error:
                        print(f"Error processing batch: {str(batch_error)}")
                        print("Trying to process documents one by one...")
                        
                        # Jika terjadi kesalahan pada batch, coba proses satu per satu
                        for doc in batch:
                            try:
                                # Pastikan metadata tidak terlalu besar sebelum upload
                                metadata_size = len(json.dumps(doc.metadata).encode('utf-8'))
                                if metadata_size > 40000:  # Sedikit di bawah batas 40960
                                    print(f"Metadata too large ({metadata_size} bytes). Creating minimal version.")
                                    doc.metadata = {"url": url}
                                
                                PineconeVectorStore.from_documents(
                                    [doc],
                                    embeddings,
                                    index_name=os.getenv("PINECONE_INDEX_NAME"),
                                )
                                total_documents_processed += 1
                                print(f"Successfully processed document with size {len(doc.page_content)} chars")
                            except Exception as doc_error:
                                print(f"Failed to process a document: {str(doc_error)}")
                                continue
                
        except Exception as e:
            print(f"Error processing {url}: {str(e)}")
            continue
    
    print(f"Total documents processed and stored: {total_documents_processed}")
    print("Vectorstore update completed.")


if __name__ == "__main__":
    ingest_docs_firecrawl()
