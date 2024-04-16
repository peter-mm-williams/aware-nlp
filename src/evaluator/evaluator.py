from src.retriever import BaseRetriever, ChromaRetriever, QdrantRetriever, CustomRetriever
from typing import Literal, List, Dict, Any
from langchain_core.documents import Document
from sklearn.metrics import recall_score, f1_score, precision_score
from langchain_community.document_loaders import DataFrameLoader
import numpy as np
import pandas as pd

class RetrieverEvaluator:
    """
    Class which evaluates the quality of retrieval of a given retriever pipeline on a given dataset

    Construction Arguments:
        1. retriever_name: Defines the pipeline that will be utilized. One of the following:
            - 'chroma' : evaluates on ChromaRetriever class object
            - 'qdrant' : evaluates on QdrantRetriever class object
            - 'custom' : evaluates on CustomRetriever class object
        2. encoder_name: embedding model to be used. Makes use of models found at:
            - [SBERT](https://www.sbert.net/docs/pretrained_models.html)
            - [HuggingFace](https://huggingface.co/spaces/mteb/leaderboard)
        3. similarity_metric: similarity metric used to rank closest documents to the query in the 
        embedding space. One of the following:
            - 'cosine-similarity'
            - 'dot-product'
            - 'euclidian'
            - 'manhattan' (only available for qdrant retriever)
        4. sample_df: Dataframe that documents were sourced from. 
        Required to have the following columns to work:
            - 'question' : query for which we are assessing the quality of retrieval
            - 'statement' : statements that will be formatted as documents and filled into a database
            - 'average_label' : the fraction of human labelers who have labeled the 
            statement as relevant to the question
            - 'label_sum' : the total number of human labelers who have labeled the 
            statement as relevant to the question
        5. retrieved_doc_size: The number of documents to be retrieved from which evaluation scores will be calculated
        6. consensus_threshold: The minimum number of 'label_sum' for which a document is considered "True"

    Calling method: evaluate()
        1. Actions:
            - Defines the consensus label based on the consensus_threshold
            - Constructs the retriever class object
            - Loads documents
            - Calculates f1, precision, and recall scores of the retrieval
        2. Outputs: A dataframe containing the following columns:
            - 'question': question for which the retrieval quality was evaluated 
            - 'retriever': name of retriever pipeline 
            - 'encoder': name of embedding model
            - 'similarity': name of similarity metric used,
            - 'consensus_threshold': The minimum number of human labelers for which a document is considered ,
            - 'retrieval_size': Number of documents retrieved per question
            - 'total_true_labels': Total number of true labels in the set of documents for the given 
            question as defined by the consensus threshold
            - 'f1': calculated f1 score for the given question
            - 'recall': calculated recall score for the given question
            - 'precision': calculated precision score for the given question
    """
    
    def __init__(
            self, 
            retriever_name: Literal['chroma', 'qdrant', 'custom'], 
            encoder_name: str,
            similarity_metric: str,
            sample_df: pd.DataFrame, 
            retrieved_doc_size: int = 10,
            consensus_threshold: int = 2
        ):
        self.retriever_name = retriever_name
        self.consensus_threshold = consensus_threshold
        self.sample_df = sample_df
        self.set_consensus()
        self.similarity_name = similarity_metric
        self.encoder_name = encoder_name
        self.n_retrieved = retrieved_doc_size
        self.questions = sample_df.question.unique()
        self.dataset_dict = self.parse_dataset()

    def set_consensus(self):
        self.sample_df['consensus'] = (self.sample_df['label_sum'] >= self.consensus_threshold).astype(int)

    def get_retriever(self, docs: List[Document]) -> BaseRetriever:
        match self.retriever_name:
            case 'chroma':
                return ChromaRetriever(docs, self.encoder_name, self.similarity_name)
            case 'qdrant':
                return QdrantRetriever(docs, self.encoder_name, self.similarity_name)
            case 'custom':
                return CustomRetriever(docs, self.encoder_name, self.similarity_name)
            case _:
                raise ValueError(f"Retriever {self.retriever_name} not supported")

    def _parse_dataset_one_question(self, question: str):
        df = self.sample_df.drop(columns = ['label_sum', 'average_label', 'consensus']).copy()
        df = df[df.question==question].fillna('')
        loader = DataFrameLoader(df, page_content_column='statement')
        return loader.load()
        
    def parse_dataset(self):
        return {
            q:self._parse_dataset_one_question(q) 
            for q in self.questions
        }

    def retrieve_one_question(self, question: str):
        retriever = self.get_retriever(self.dataset_dict[question])
        retriever.load_documents()
        retrieved_docs = retriever.retrieve(question, n = self.n_retrieved)
        return retrieved_docs
    
    def _parse_retriever_element(self, elem: Dict[str, Any]):
        return {'statement':elem['document'].page_content, 'score':elem['score']} | elem['document'].metadata

    def _construct_evaluation_df(self, question: str, output: List[Dict[str, Any]]) -> pd.DataFrame:
        df_pred = pd.DataFrame([self._parse_retriever_element(elem) for elem in output])
        df_pred['retrieved'] = 1
        return self.sample_df[self.sample_df.question==question].merge(
            df_pred, how='outer').fillna({'retrieved':0, 'consensus':0})

    def evaluate_one_question(self, question: str):
        retrieved_docs = self.retrieve_one_question(question)
        eval_df = self._construct_evaluation_df(question, retrieved_docs)

        f1 = f1_score(eval_df['consensus'], eval_df['retrieved'])
        recall = recall_score(eval_df['consensus'], eval_df['retrieved'])
        precision = precision_score(eval_df['consensus'], eval_df['retrieved'])
        
        return {
            'question':question, 
            'retriever':self.retriever_name, 
            'encoder':self.encoder_name,
            'similarity':self.similarity_name,
            'consensus_threshold':self.consensus_threshold,
            'retrieval_size': self.n_retrieved,
            'total_true_labels': eval_df['consensus'].sum(),
            'f1':f1, 
            'recall':recall, 
            'precision':precision
        }

    def evaluate(self):
        return pd.DataFrame([self.evaluate_one_question(q) for q in self.questions])