from langchain_chroma import Chroma
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os
from dotenv import load_dotenv
load_dotenv()

CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")
CHROMA_DB_COLLECTION_NAME = os.getenv("CHROMA_DB_COLLECTION_NAME", "genomic_curation_collection")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")

def make_tsne_visualization(chroma_db_path: str = CHROMA_DB_PATH, chroma_db_collection_name: str = CHROMA_DB_COLLECTION_NAME) -> None:
    # load Chroma vector store
    chroma_db = Chroma(
        persist_directory=chroma_db_path,
        collection_name=chroma_db_collection_name
    )
    # load embedding and metadata
    all_data = chroma_db._collection.get(include=["embeddings", "metadatas"])
    embeddings = all_data["embeddings"]
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_tsne = tsne.fit_transform(embeddings)
    plt.scatter(embeddings_tsne[:, 0], embeddings_tsne[:, 1], alpha=0.6)
    plt.title("t-SNE Visualization of Document Embeddings")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.grid(True)
    plt.show()