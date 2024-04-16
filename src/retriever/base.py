from langchain_core.documents import Document
from typing import List
from abc import ABC, abstractmethod
from sentence_transformers import SentenceTransformer


class BaseRetriever(ABC):
    """
    Abstract Base Class for Retriever objects for use in a RAG pipeline. 
    To be inherited by specific retriever implementations.

    Inputs:
        - documents: List of langchain Document objects
        - embedding_model: name of the SentenceTransformer model to be used for encoding
        - similarity_metric: name of the similarity metric to be used
    
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
        - This instantiates the in-memory database, computes embeddings, and 
        loads them into the database
    2. retrieve(query, n=n_docs)
        - Retrieves the n_docs documents with the best similarity scores in 
        accordance to the metric provided at object construction
    3. make_embeddings()
        - Computes the embeddings for the documents in the database and assigns 
        them to the "embeddings" attribute
    
    """

    def __init__(
        self,
        documents: List[Document],
        embedding_model: str,
        similarity_metric: str,
        content_field: str = 'page_content'
    ):
        self.documents = documents
        self.embedding_name = embedding_model
        self.similarity_name = similarity_metric
        self.content_field = content_field
        self.encoder = None
        self.embeddings = None

    def _instantiate_embedding_model(self):
        self.encoder = SentenceTransformer(self.embedding_name)

    @abstractmethod
    def load_documents(self):
        """ Method to instantiate db and load documents into memory"""
        pass

    def make_embeddings(self):
        """ Embed documents content field and assign as a list to the attribute embeddings"""
        self._instantiate_embedding_model()
        self.embeddings = [self.encoder.encode(
            getattr(doc, self.content_field)) for doc in self.documents]

    @abstractmethod
    def retrieve(self, query: str, n: int = 10) -> List[Document]:
        """ 
        Method to retrieve the top n documents based on similarity to the 
            query in embedded space according to the similarity metric specified in the __init__
        """
        pass
