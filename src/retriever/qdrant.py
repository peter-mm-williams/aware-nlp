from typing import Optional, List, Dict, Any, Literal
from qdrant_client import models, QdrantClient
from src.retriever import BaseRetriever
from langchain_core.documents import Document


class QdrantRetriever(BaseRetriever):

    def __init__(
        self,
        documents: List[Document],
        embedding_model: str,
        similarity_metric: Literal['cosine-similarity', 'dot-product', 'euclidian', 'manhattan'],
        content_field: str = 'page_content',
        collection_name: str = 'test-collection'
    ):
        super().__init__(documents, embedding_model, similarity_metric, content_field)
        self.collection_name = collection_name
        self.name = 'qdrant'

    def _instantiate_metric(self):
        match self.similarity_name:
            case 'cosine-similarity':
                self.metric = models.Distance.COSINE
            case 'dot-product':
                self.metric = models.Distance.DOT
            case 'euclidian':
                self.metric = models.Distance.EUCLID
            case 'manhattan':
                self.metric = models.Distance.MANHATTAN
            case _:
                self.metric = models.Distance.COSINE

    def _instantiate_db(self):
        self._instantiate_metric()
        self.db = QdrantClient(":memory:")
        self.db.create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(
                # Vector size is defined by used model
                size=self.encoder.get_sentence_embedding_dimension(),
                distance=self.metric
            )
        )

    def load_documents(self):
        if self.embeddings is None:
            self.make_embeddings()
        self._instantiate_db()
        self.db.upload_points(
            collection_name=self.collection_name,
            points=[
                models.PointStruct(
                    id=idx,
                    vector=embedding.tolist(),
                    payload=doc.__dict__,
                ) for idx, (doc, embedding) in enumerate(zip(self.documents, self.embeddings))
            ]
        )

    def retrieve(self, query: str, n: int = 10) -> List[Dict[str, Any]]:
        hits = self.db.search(
            collection_name=self.collection_name,
            query_vector=self.encoder.encode(query).tolist(),
            limit=n
        )
        return [
            {
                'document': Document(**hit.payload),
                'score': hit.score
            }
            for hit in hits
        ]
