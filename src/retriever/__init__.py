from src.retriever.retriever import Retriever
from typing import Callable
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import numpy as np


def load_retriever(
    pickle_filename: str,
    embedding_filename: str = None,
    similarity_metric: Callable = cosine_similarity
) -> Retriever:
    with open(pickle_filename, 'rb') as handle:
        ret_dict = pickle.load(handle)

    ret = Retriever(
        model_name=ret_dict['model_name'],
        reddit_ids=ret_dict['reddit_ids'],
        sentences=ret_dict['sentences'],
        similarity_metric=similarity_metric,
        chunk_overlap=ret_dict['chunk_overlap']
    )
    ret.chunks = ret_dict['chunks']
    ret.chunk_ids = ret_dict['chunk_ids']
    if embedding_filename:
        with open(embedding_filename, 'r') as handle:
            embeddings = np.load(embedding_filename)
        ret.embeddings = embeddings
    return ret
