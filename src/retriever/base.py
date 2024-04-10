from langchain_core.documents import Document
from typing import List
from abc import ABC, abstractmethod
from sentence_transformers import SentenceTransformer


class BaseRetriever(ABC):

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
