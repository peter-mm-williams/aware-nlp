from langchain_core.documents import Document
from typing import List, Optional
from abc import ABC, abstractmethod
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter


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
        content_field: str = 'page_content',
        chunk_overlap: int = 50
    ):
        self.documents = documents
        self.embedding_name = embedding_model
        self.similarity_name = similarity_metric
        self.content_field = content_field
        self.chunk_overlap = chunk_overlap
        self.splitter = None
        self.encoder = None
        self.embeddings = None

    def _initialize_embedding_model(self):
        if self.encoder is None:
            self.encoder = SentenceTransformer(self.embedding_name)

    def _sentence_token_length(self, sentence: str):
        return len(self.encoder.tokenizer.tokenize(sentence))

    def _initialize_splitter(self, chunk_overlap: Optional[int] = None):
        self._initialize_embedding_model()
        if chunk_overlap is not None:
            self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.encoder.max_seq_length,
            chunk_overlap=self.chunk_overlap,
            length_function=self._sentence_token_length
        )

    def split_documents(self, chunk_overlap: Optional[int] = None):
        self._initialize_splitter(chunk_overlap=chunk_overlap)
        # Run split_setence on each sentence and fill chunk_ids, chunks attributes
        self.documents = self.splitter.split_documents(self.documents)

    @abstractmethod
    def load_documents(self):
        """ Method to instantiate db and load documents into memory"""
        pass

    def make_embeddings(self):
        """ Embed documents content field and assign as a list to the attribute embeddings"""
        self._initialize_embedding_model()
        self.embeddings = [self.encoder.encode(
            getattr(doc, self.content_field)) for doc in self.documents]

    @abstractmethod
    def retrieve(self, query: str, n: int = 10) -> List[Document]:
        """ 
        Method to retrieve the top n documents based on similarity to the 
            query in embedded space according to the similarity metric specified in the __init__
        """
        pass
