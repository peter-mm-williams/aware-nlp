import numpy as np
from langchain_core.documents import Document
from sentence_transformers import util, SentenceTransformer
from typing import List, Literal, Any, Dict
from sklearn.metrics.pairwise import euclidean_distances
from src.retriever import BaseRetriever


class ClusterRetriever(BaseRetriever):

    def __init__(
        self,
        documents: List[Document],
        embedding_model: str,
        similarity_metric: Literal['cosine-similarity', 'dot-product', 'euclidian'],
        content_field: str = 'page_content',
        clustering_model = None
    ):
        super().__init__(documents, embedding_model, similarity_metric, content_field)
        self.clustering_model = clustering_model
        self.name = 'custom'

    def load_documents(self):
        self.make_embeddings()

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

    def _calc_similarity_score(self, query_embedding) -> np.array:
        scores = self.metric(query_embedding, self.embeddings)[0]
        if self.similarity_name == 'euclidian':
            return scores
        return scores.numpy()

    def retrieve(self, query: str, n: int = 10) -> List[Dict[str, Any]]:
        self._instantiate_metric()
        if self.embeddings is None:
            self.make_embeddings()
        query_embedding = self.encoder.encode([query])
        scores = self._calc_similarity_score(query_embedding)
        sorted_idxs = np.argsort(scores)
        if self.similarity_name != 'euclidian':
            sorted_idxs = sorted_idxs[::-1]
        
        # get clusters present in top 5
        if self.clustering_model is None:
            raise ValueError('No clustering model provided.')
        cluster_labels = self.clustering_model.labels_
        top_clusters = set(cluster_labels[i] for i in sorted_idxs[:5])
        # prioritize retrieving documents from clusters present in top 5 clusters
        top_cluster_idxs = []
        other_idxs = []
        for i in sorted_idxs:
            if cluster_labels[i] in top_clusters:
                top_cluster_idxs.append(i)
            else:
                other_idxs.append(i)
        sorted_idxs = top_cluster_idxs + other_idxs

        return [
            {
                'document': self.documents[i],
                'score': scores[i]
            }
            for i in sorted_idxs[:n]
        ]
