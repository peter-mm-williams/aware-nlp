import os

import json
import pandas as pd
from langchain_community.document_loaders import DataFrameLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_mistralai import MistralAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings import OpenAIEmbeddings

from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores import Chroma

from langchain_community.llms import HuggingFaceEndpoint
from langchain_core.prompts import ChatPromptTemplate
from langchain.prompts import PromptTemplate

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_groq import ChatGroq


path = os.getcwd()

class BasicRAG:
    def __init__(self, dataset, query, llm_model, api_key, search_type, kwarg_k, score_threshold, gen_prompt):
        self.dataset = dataset
        self.query=query
        self.llm_model = llm_model
        self.api_key = api_key
        self.search_type = search_type
        self.kwarg_k = kwarg_k
        self.score_threshold = score_threshold
        self.gen_prompt = gen_prompt
        self.df_subreddit = self.set_subreddit()
    
    def set_subreddit(self):
        with open(path+"/"+self.dataset+'.json', 'r', encoding='utf-8') as f:
            dat = json.load(f)
        return pd.DataFrame(dat)

    def rag_response(self):
        persist_directory = path + '/WholeVS/'
        embeddings = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")
        vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

        retriever = vectorstore.as_retriever(search_kwargs={"k": self.kwarg_k})

        model = ChatGroq(temperature=0, groq_api_key=self.api_key, model_name=self.llm_model)

        prompt = ChatPromptTemplate.from_template(self.gen_prompt)

        document_chain = create_stuff_documents_chain(model, prompt)
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        return retrieval_chain.invoke({"input": self.query})