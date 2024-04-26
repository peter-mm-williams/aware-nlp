import numpy as np
import pandas as pd
from langchain_core.documents import Document
from sentence_transformers import util
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
        clustering_model=None,
        all_chunks=None
    ):
        super().__init__(documents, embedding_model, similarity_metric, content_field)
        self.clustering_model = clustering_model
        self.name = 'cluster'
        if all_chunks is None:
            chunks_file = '../../data/best_buy/chunks.csv'
            df = pd.read_csv(chunks_file)
            self.all_chunks = df['chunks']

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
        top_idxs_in_all_chunks = []
        all_chunks_list = self.all_chunks.to_list()
        for i in sorted_idxs[:5]:
            doc = getattr(self.documents[i], self.content_field)
            if doc in all_chunks_list:
                top_idxs_in_all_chunks.append(all_chunks_list.index(doc))
        top_clusters = set(cluster_labels[j] for j in top_idxs_in_all_chunks)
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


class ClusterRetrieverCentroid(BaseRetriever):

    def __init__(
        self,
        documents: List[Document],
        embedding_model: str,
        similarity_metric: Literal['cosine-similarity', 'dot-product', 'euclidian'],
        content_field: str = 'page_content',
        clusters: List[int] = None,
        cluster_docs: List[int] = None,
        clustered_embeddings: np.array = None
    ):
        super().__init__(documents, embedding_model, similarity_metric, content_field)
        self.name = 'cluster'
        self.clusters = clusters
        self.cluster_docs = cluster_docs
        self.clustered_embeddings = clustered_embeddings
        self.centroids = None
        self.doc_cluster_labels = None

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

    def _set_centroids(self):
        unique_labels = np.sort(np.unique(self.clusters))
        self.centroids = np.array([np.mean(
            self.clustered_embeddings[self.clusters == i, :], axis=0) for i in unique_labels])

    def _label_docs_clusters(self):
        chunk_ids = [self.cluster_docs.index(doc) for doc in self.documents]
        self.doc_cluster_labels = self.clusters[chunk_ids]

    def get_sorted_clusters(self, query_embedding):
        if self.centroids is None:
            self._set_centroids()
        if self.doc_cluster_labels is None:
            self._label_docs_clusters()
        cluster_scores = self.metric(query_embedding, self.centroids)[0]
        if self.similarity_name != 'euclidian':
            cluster_scores = cluster_scores.numpy()
            # reverse order to sort in descending order of score
            return np.argsort(-cluster_scores)
        return np.argsort(cluster_scores)

    def compare_score(self, score: float, distance_cutoff: float):
        if self.similarity_name == 'euclidian':
            return score < distance_cutoff
        return score > distance_cutoff

    def retrieve_docs_from_cluster(self, query_embedding: np.array, cluster_label: int, distance_cutoff: float, max_n: int):
        cluster_docs = np.array(self.documents)[
            self.doc_cluster_labels == cluster_label]
        if len(cluster_docs) == 0:
            return []
        cluster_embeddings = np.array(self.embeddings)[
            self.doc_cluster_labels == cluster_label, :]
        scores = self.metric(query_embedding, cluster_embeddings)[0]
        if self.similarity_name != 'euclidian':
            scores = scores.numpy()
            sorted_idxs = np.argsort(-scores)
        else:
            sorted_idxs = np.argsort(scores)
        return [
            {
                'document': cluster_docs[i],
                'score': scores[i]
            }
            for i in sorted_idxs[:max_n] if self.compare_score(scores[i], distance_cutoff)
        ]

    def retrieve(self, query: str, distance_cutoff: float, n: int = 10) -> List[Dict[str, Any]]:
        self._instantiate_metric()
        if self.embeddings is None:
            self.make_embeddings()
        query_embedding = self.encoder.encode([query])

        retrieved_docs = []
        sorted_clusters = self.get_sorted_clusters(query_embedding)

        # Loop over clusters retrieved the top docs better than the cutoff
        for cluster_id in sorted_clusters:
            # print(f'Searching in cluster {cluster_id}')
            retrieved_docs += self.retrieve_docs_from_cluster(
                query_embedding,
                cluster_id,
                distance_cutoff,
                n - len(retrieved_docs)
            )
            if len(retrieved_docs) == n:
                break
        return retrieved_docs
