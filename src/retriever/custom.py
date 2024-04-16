import numpy as np
from langchain_core.documents import Document
from sentence_transformers import util, SentenceTransformer
from typing import List, Literal, Any, Dict
from sklearn.metrics.pairwise import euclidean_distances
from src.retriever import BaseRetriever


class CustomRetriever(BaseRetriever):
    """
    Custom Retriever class for use in a RAG pipeline. Takes in a list of langchain Document objects, 
    an embedding model, and a similarity metric to be used for retrieval. Computes embeddings for the
    content field of the Document objects, stores as a list of numpy arrays as the "embeddings" attribute,
    and retrieves documents based on the similarity metric provided at object construction. Makes use of
    the SentenceTransformer library for encoding and similarity calculations (uses sklearn for euclidian distance).

    Inputs:
        - documents: List of langchain Document objects
        - embedding_model: name of the SentenceTransformer model to be used for encoding
        - similarity_metric: name of the similarity metric to be used
        - content_field: name of the field in the Document object to be used for retrieval

    Output:
        - List of dictionaries containing the document object and the similarity score to the query
    
    Attributes:
        - documents: List of langchain Document objects containing the content 
        to be considered for retieval
        - embedding_name: name of the SentenceTransformer model to be used for encoding
        - similarity_name: name of the similarity metric to be used 
        to determine similarity between documents and queries
        - content_field: name of the field in the Document object to be used for retrieval
        - encoder: SentenceTransformer object to be used to encode the 
        content_field of the documents
        - embeddings: List of embeddings corresponding to the content field of the documents
    
    Methods:
    1. load_documents()
        - Computes embeddings and assigns them to the "embeddings" attribute
    2. _calc_similarity_score(query_embedding)
        - Calculates the similarity scores between the query embedding and the document embeddings
    3. retrieve(query, n=n_docs)
        - Retrieves the n_docs documents with the best similarity scores in 
        accordance to the metric provided at object construction
    """

    def __init__(
        self,
        documents: List[Document],
        embedding_model: str,
        similarity_metric: Literal['cosine-similarity', 'dot-product', 'euclidian'],
        content_field: str = 'page_content'
    ):
        super().__init__(documents, embedding_model, similarity_metric, content_field)
        self.name = 'custom'

    def load_documents(self):
        self.make_embeddings()

    def _instantiate_metric(self):
        # Assigns the appropriate similarity metric to the "metric" attribute based on the similarity_name
        match self.similarity_name:
            case 'cosine-similarity':
                self.metric = util.cos_sim
            case 'dot-product':
                self.metric = util.dot_score
            case 'euclidian':
                self.metric = euclidean_distances
            case _:
                self.metric = util.cos_sim

    def _calc_similarity_score(self, query_embedding) -> np.array:
        # Calculates the similarity scores between the query embedding and the document embeddings
        scores = self.metric(query_embedding, self.embeddings)[0]
        if self.similarity_name == 'euclidian':
            return scores
        return scores.numpy()

    def retrieve(self, query: str, n: int = 10) -> List[Dict[str, Any]]:
        # Retrieves the n_docs documents with the best similarity scores
        self._instantiate_metric()
        if self.embeddings is None:
            self.make_embeddings()
        query_embedding = self.encoder.encode([query])
        scores = self._calc_similarity_score(query_embedding)
        sorted_idxs = np.argsort(scores)
        if self.similarity_name != 'euclidian':
            sorted_idxs = sorted_idxs[::-1]
        return [
            {
                'document': self.documents[i],
                'score': scores[i]
            }
            for i in sorted_idxs[:n]
        ]
