from typing import Optional, List, Dict, Any, Literal
from langchain_community.vectorstores import Chroma
from src.retriever import BaseRetriever
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import euclidean_distances
from sentence_transformers import util


class ChromaRetriever(BaseRetriever):

    def __init__(
        self,
        documents: List[Document],
        embedding_model: str,
        similarity_metric: Literal['cosine-similarity', 'dot-product', 'euclidian'],
        content_field: str = 'page_content',
        collection_name: str = 'test-collection'
    ):
        super().__init__(documents, embedding_model, similarity_metric, content_field)
        self.collection_name = collection_name
        self.name = 'qdrant'
        self.db = None

    def _instantiate_embedding_model(self):
        self.encoder = HuggingFaceEmbeddings(model_name=self.embedding_name)

    @property
    def metric_abbreviation(self):
        abbrev_dict = {
            'cosine-similarity': "cosine",
            'dot-product': 'ip',
            'euclidian': 'l2'
        }
        return abbrev_dict.get(self.similarity_name, 'cosine')

    def _instantiate_metric(self):
        match self.similarity_name:
            case 'cosine-similarity':
                self.metric = util.cos_sim
            case 'dot-product':
                self.metric = util.dot_score
            case 'euclidian':
                self.metric = euclidean_distances
            case _:
                self.metric = util.cos_sim

    def load_documents(self):
        self._instantiate_embedding_model()
        self._instantiate_metric()
        self.db = Chroma.from_documents(
            documents=self.documents,
            embedding=self.encoder,
            collection_metadata={"hnsw:space": self.metric_abbreviation},
            relevance_score_fn=self.metric
        )

    def retrieve(self, query: str, n: int = 10) -> List[Dict[str, Any]]:
        hits = self.db.similarity_search_with_score(
            query=query,
            k=n
        )
        return [
            {
                'document': doc,
                'score': score
            }
            for (doc, score) in hits
        ]
