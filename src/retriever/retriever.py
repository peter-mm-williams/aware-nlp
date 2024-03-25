from typing import List, Callable
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pickle


class Retriever:
    """
    Given the inputs below performs the retrieval step of RAG:
    - model_name: for the embeddings (see: https://www.sbert.net/docs/pretrained_models.html)
    - sentences: a list of strings to be embedded
    - similarity_metric: a function to evaluate similarities (defaults to cosine_similarity)
    - query: prompt against which to retrieve
    """

    def __init__(self,
                 model_name: str,
                 reddit_ids: List[str],
                 sentences: List[str],
                 similarity_metric: Callable = cosine_similarity,
                 chunk_overlap: int = 20
                 ):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.chunk_overlap = chunk_overlap
        self.reddit_ids = reddit_ids
        self.sentences = sentences
        self.metric = similarity_metric
        self.chunk_ids = []
        self.chunks = []
        self.embeddings = None

    def _sentence_token_length(self, sentence: str):
        return len(self.model.tokenizer.tokenize(sentence))

    def split_sentence(self, sentence: str):
        # Split sentences into chunks such that each chunk is less than the max token length of the model

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.model.max_seq_length,
            chunk_overlap=20,
            length_function=self._sentence_token_length
        )
        return splitter.split_text(sentence)

    def make_chunks(self):
        # Run split_setence on each sentence and fill chunk_ids, chunks attributes
        for reddit_id, sentence in zip(self.reddit_ids, self.sentences):
            chunks = self.split_sentence(sentence)
            self.chunk_ids += [reddit_id] * len(chunks)
            self.chunks += chunks

    def make_embeddings(self):
        # sentences to embedding space; fills attribute "embeddings"
        if len(self.chunks) == 0:
            self.make_chunks()
        self.embeddings = self.model.encode(self.sentences)

    def _save_embeddings(self, filename: str):
        # save embeddings to a npy file
        np.save(filename, self.embeddings)

    def _save_attributes(self, filename: str):
        data = {
            'model_name': self.model_name,
            'reddit_ids': self.reddit_ids,
            'sentences': self.sentences,
            'chunk_ids': self.chunk_ids,
            'chunks': self.chunks,
            'chunk_overlap': self.chunk_overlap,
        }
        with open(filename, 'wb') as f:
            # write retriever attributes to pickle file
            pickle.dump(data, f)

    def save(self, attribute_filename: str, embedding_filename: str = None):
        # save retriever data to pickle file
        self._save_attributes(attribute_filename)
        if self.embeddings:
            self._save_embeddings(embedding_filename)

    def retrieve(self, query: str, n: int = 10):
        # finds the n closest sentences in embedding space according to the provided similarity_metric

        if self.embeddings is None:
            self.make_embeddings()
        query_embedding = self.model.encode([query])
        scores = self.metric(query_embedding, self.embeddings)
        top_n_idx = np.argsort(scores[0])[::-1][:n]
        return [
            {
                'response': self.sentences[i],
                'score': scores[0][i]
            }
            for i in top_n_idx
        ]
