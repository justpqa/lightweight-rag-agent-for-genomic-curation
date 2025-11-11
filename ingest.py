from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import fitz
import os
import re
from dotenv import load_dotenv
load_dotenv()

CORPUS_PATH = os.getenv("CORPUS_PATH", "./corpus")
CHROMA_DB_COLLECTION_NAME = os.getenv("CHROMA_DB_COLLECTION_NAME", "genomic_curation_collection")
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 500))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 50))

def clean_text(text: str) -> str:
    # Lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    # Remove tabs and newlines
    text = text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
    # Remove special characters and punctuation (keep basic ones if needed)
    text = re.sub(r'[^a-z0-9\s\.\,\-]', '', text)
    # Replace multiple spaces with single space
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def clean_document(document: Document) -> Document:
    cleaned_text = clean_text(document.page_content)
    cleaned_document = Document(page_content=cleaned_text, metadata=document.metadata)
    return cleaned_document

def ingest_corpus(chroma_db_collection_name: str = CHROMA_DB_COLLECTION_NAME) -> None:
    # load documents from corpus
    print("Loading documents from corpus...")    
    documents = []
    metadata = [] # store a list of metadata dictionaries, currenly only have filenames
    for filename in os.listdir(CORPUS_PATH):
        if filename.endswith(".txt"):
            with open(os.path.join(CORPUS_PATH, filename), 'r', encoding='utf-8') as f:
                documents.append(f.read())
                metadata.append({"filename": filename})
        elif filename.endswith(".pdf"):
            pdf_path = os.path.join(CORPUS_PATH, filename)
            with fitz.open(pdf_path) as doc:
                text = ""
                for page in doc:
                    text += page.get_text() + "\n\n" # special indicator of pages
                documents.append(text)
                metadata.append({"filename": filename})
    print(f"Finished loading {len(documents)} documents.")
    print()

    # split documents into chunks
    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, 
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""]
    )
    splitted_documents = text_splitter.create_documents(texts=documents, metadatas=metadata)
    splitted_documents = [clean_document(doc) for doc in splitted_documents]
    print(f"Finished splitting to make {len(splitted_documents)} chunks.")
    print()

    # create Chroma vector store
    print("Creating Chroma vector store...")
    chroma_db = Chroma(
        persist_directory=CHROMA_DB_PATH,
        embedding_function=HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME),
        collection_name=chroma_db_collection_name
    )
    # delete current collection contents before adding new documents
    collection = chroma_db._collection
    all_docs = collection.get(include=[])
    all_ids = all_docs["ids"]
    if all_ids:
        collection.delete(ids=all_ids)
    # add documents to Chroma vector store
    chroma_db.add_documents(splitted_documents)
    print("Finished creating Chroma vector store.")
    print()