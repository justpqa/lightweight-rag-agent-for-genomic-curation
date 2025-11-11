from typing import Optional, List, Tuple
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import CrossEncoder
import os
from dotenv import load_dotenv
load_dotenv()

CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")
CHROMA_DB_COLLECTION_NAME = os.getenv("CHROMA_DB_COLLECTION_NAME", "genomic_curation_collection")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "ibm-granite/granite-4.0-h-350m")
RERANKER_MODEL_NAME = os.getenv("RERANKER_MODEL_NAME", "cross-encoder/ms-marco-MiniLM-L-6-v2")
PROMPT_TEMPLATE_PATH = os.getenv("PROMPT_TEMPLATE_PATH", "./prompts/prompt_v1.txt")
TOP_K = int(os.getenv("TOP_K", 5))
TOP_K_RERANK = int(os.getenv("TOP_K_RERANK", 3))
SIMILARITY_SCORE_THRESHOLD = float(os.getenv("SIMILARITY_SCORE_THRESHOLD", 0.0))
NUM_TRIALS = int(os.getenv("NUM_TRIALS", 2))
TEMPERATURE = float(os.getenv("TEMPERATURE", 0.3))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", 250))

class RAG:
    def __init__(self, chroma_db_path: str = CHROMA_DB_PATH, chroma_db_collection_name: str = CHROMA_DB_COLLECTION_NAME,
                 embedding_model_name: str = EMBEDDING_MODEL_NAME, reranker_model_name: str = RERANKER_MODEL_NAME, 
                 llm_model_name: str = LLM_MODEL_NAME, prompt_template_path: str = PROMPT_TEMPLATE_PATH, 
                 num_trials: int = NUM_TRIALS, top_k: int = TOP_K, top_k_rerank: int = TOP_K_RERANK, 
                 similarity_score_threshold: float = SIMILARITY_SCORE_THRESHOLD, temperature: float = TEMPERATURE, 
                 max_tokens: Optional[int] = MAX_TOKENS, print_progress: bool = True) -> None:
        if print_progress:
            print("Initializing RAG system...") 
            print()

        # vector store
        if print_progress:
            print("Loading vector store...")
        self.vector_store = Chroma(
            persist_directory=chroma_db_path,
            embedding_function=HuggingFaceEmbeddings(model_name=embedding_model_name),
            collection_name=chroma_db_collection_name
        )
        if print_progress:
            print("Finished loading vector store.")
            print()

        # LLM
        # we will use primaritly CPU for LLM inference, 
        # adjust as needed by changing to device = 0, 1, etc. for GPU, -1 for CPU
        if print_progress:
            print("Loading LLM...")
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
        self.model = AutoModelForCausalLM.from_pretrained(llm_model_name)
        self.model.eval()
        if print_progress:
            print("Finished loading LLM.")
            print()

        if print_progress:
            print("Loading reranker model...")
        self.reranker_model = CrossEncoder(
            model_name_or_path=reranker_model_name, 
            device="cpu"
        )
        if print_progress:
            print("Finished loading reranker model.")
            print()

        # load prompt template
        if print_progress:
            print("Loading prompt template...")
        self.prompt_template = ""
        with open(prompt_template_path, 'r') as f:
            self.prompt_template = f.read()
        if print_progress:
            print("Finished loading prompt template.")
            print()

        # params for retrieval
        self.num_trials = num_trials
        self.top_k = top_k
        self.top_k_rerank = top_k_rerank
        self.similarity_score_threshold = similarity_score_threshold
        self.max_tokens = max_tokens
        self.temparature = temperature

        if print_progress:
            print("RAG system initialized.")

    def retrieve(self, query: str) -> List[Document]:
        """Retrieve relevant documents from the vector store."""
        retrieved_documents = self.vector_store.similarity_search_with_relevance_scores(query = query, k=self.top_k)
        
        # filter by similarity score threshold
        filtered_documents = [doc for doc, score in retrieved_documents if score >= self.similarity_score_threshold]
        return filtered_documents
    
    def retrieve_with_scores(self, query: str) -> List[Tuple[int, Document]]:
        """Retrieve relevant documents along with their similarity scores from the vector store."""
        retrieved_documents = self.vector_store.similarity_search_with_relevance_scores(query = query, k=self.top_k)
        
        # filter by similarity score threshold
        filtered_documents_with_score = [(score, doc) for doc, score in retrieved_documents if score >= self.similarity_score_threshold]
        return filtered_documents_with_score
    
    def rerank(self, query: str, documents: List[Document]) -> List[Document]:
        """Rerank the retrieved documents using the cross-encoder reranker."""
        scores = self.reranker_model.predict([(query, doc.page_content) for doc in documents])
        reranked_docs = [doc for _, doc in sorted(zip(scores, documents), key=lambda x: x[0], reverse=True)]
        return reranked_docs[:self.top_k_rerank]
    
    def generate(self, prompt: str) -> str:
        input_tokens = self.tokenizer(prompt, return_tensors="pt")
        # generate output tokens
        output = self.model.generate(**input_tokens, 
                                     max_new_tokens = self.max_tokens,
                                     temperature = self.temparature,
                                     do_sample=True)
        # decode output tokens into text
        output = self.tokenizer.batch_decode(output)[0]
        output = output.split("<|start_of_role|>assistant<|end_of_role|>")[-1].replace("<|end_of_text|>", "").strip()
        return output
    
    def format_answer_with_citations(self, answer: str, documents: List[Document]) -> str:
        """Format the answer along with citations to the source documents."""
        formatted_citations = "\n\n".join([f"Citation {i+1} ({doc.metadata.get('filename', '')}):\n{doc.page_content}" 
                                           for i, doc in enumerate(documents)])
        formatted_answer = f"{answer}\n\nSources:\n{formatted_citations}"
        return formatted_answer

    def answer(self, query: str, with_citation: bool = True) -> str:
        """Generate an answer based on the retrieved documents and the query."""
        # Retrieve
        retrieved_docs = []
        for _ in range(self.num_trials):
            retrieved_docs = self.retrieve(query)
            if len(retrieved_docs) > 0:
                break
        if len(retrieved_docs) == 0:
            return "I'm sorry, I couldn't find relevant information to answer your question."
        
        # Rerank
        reranked_retrived_docs = self.rerank(query, retrieved_docs)

        # Generation
        citations = "\n".join([f"{doc.metadata.get('filename', '')}:\n{doc.page_content}" for doc in reranked_retrived_docs])
        prompt = self.prompt_template.format(
            query=query,
            citations=citations
        )
        response = self.generate(prompt)
        if with_citation:
            response = self.format_answer_with_citations(response, reranked_retrived_docs)
        return response