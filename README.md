# aware-nlp
Erdos Institute Data Science Boot Camp project partnered with Aware. This project involves the investigation and evaluation of different methodologies for retrieval for use in RAG (Retrieval-Augmented Generation) systems. In particular, this project investigates retrieval quality for information downloaded from employee subreddits.

## Project Description
#### Overview
Retrieval Augmented Generation (RAG) is a powerful approach that enhances Large Language Models (LLM) capability to generate a richer and more in-context response to user queries. By retrieving relevant information through an information retrieval system, and then generating responses, RAG ensures reliability and minimizes the risk of misinformation and hallucination.
The objective of this project is to build an information retrieval system that identifies the most relevant content in the provided dataset for a given user query and rank the retrieval content. The priorities of this project are that the retrieval is fast (occurs sub-second), and that there is a methodology to gauge the performance of the retrieval.
The information retreival procedure for a baseline RAG system involves three main stages:
- Vector indexing: The text contents will be converted to embedded vectors in a high-dimensional space using embedding models. 
- Storing in a vector store: The embedded vectors will be loaded to a vector database, where the numerical representative of our dataset can be searched efficiently for retrieval process.
- Retrieval based on similarity match: The similarity between the query and the content vectors will be calculated based on the distance between the vectors.
  
#### Evaluation Methodology

##### Dataset Pre-processing
The dataset used to evaluate these pipelines was taken from submissions and comments of a reddit thread for Best Buy employees. Statements were composed by concatenating the title and text fields of the individual submissions and comments. Timestamp, reddit_id, author, permalink, and subreddit were taken as metadata with the statements to compose documents. The documents were then split into 512 token vectors with 50 token overlaps. They were then encoded into an embedding space via a choice of embedding model. Three questions typical to the type of questions Aware’s clients would ask of the data were handwritten:
1. What do Best Buy Employees Think of the Company?
2. What are the most common reasons for Best Buy employees to leave Best Buy?
3. Do employees feel understaffed?
40 statements sampled from the reddit thread were labeled by 7 observers for each question as relevant (True) or irrelevant (False) to the question posed. This was instructive in understanding the subjectivity of the labels of the data. Setting a consensus threshold of at least 50% of the human labellers, the questions above had 10, 6, and 5 positive labels respectively. This dataset was then expanded such that 90 unique statements from the original set were labeled for each question. 

##### Automated Labeling
For the construction of larger evaluation sets, large language models (LLM) were used in the construction of larger labeled datasets. The quality of this automated labeling was generated using the “dolphin-mixtral” model and was assessed on the 7-observer labeled data. This correctly labeled 10 out of the 12 statements unanimously labeled as relevant. Using a consensus threshold of 50% of human labelers, the LLM correctly labeled all of the irrelevant statements and produced an F1 score of 0.80. 
##### Retrieval Evaluation Metrics
A procedure for evaluating different RAG pipelines on this dataset was then used to compare the quality of retrieval using different embedding models to convert statements and questions into an encoded vector space. The precision, recall, and f1 scores of the retrieved documents were calculated for a range of the number of retrieved documents. The 90 statement dataset was used to evaluate the performance of naive retrieval for a range of embedding models. “all-mpnet-base-v1” was shown to perform well for both as little as 5 retrieved documents and as many as 30 (f1 scores of 0.470.27 and 0.55 0.13, respectively). The next best model was “paraphrase-mpnet-base-v2” which had f1 scores of 0.360.22 and 0.560.18, respectively.
The Bestbuy subreddit was selected as the focus for quality evaluation.

<=================================================================================================>
#### Advanced Retrieval Methods
The impact of using clusters as a first stage indexing process (Clustering), using multiple LLM generated queries to help pinpoint the most relevant documents (Multi-query), and indexing over summaries of statements (Multi-vector) were explored in greater detail. We observe that utilizing clustering, multi-query, and multi-vector indexing techniques show slight improvement in retrieval quality. 
##### Clustering
To gain insight as to how statements are distributed in the embedded space, a LLM was used to label a set of 650 statements as to whether they contained information indicating a satisfied employee, an unsatisfied employee, or  was neutral/irrelevant. A range of clustering methods and hyperparameters were evaluated via completeness, homogeneity, v-score, and the number of clusters generated. A k-means clustering of 500 clusters on the 5,667 chunked embedded statements offered a good tradeoff between the number of clusters and performance. These clusters were then used as part of a 2-stage retrieval process by which clusters would be searched for relevant documents in order of their similarity to the query in the embedded space. This 2-stage process retrieved documents with an f1 score at or better than naive retrieval for 5, 10, 15, 20, 25, and 30 retrieved documents.
##### Multi-query
As a second advanced pipeline we explored the multi-query technique where an input query is sent into a large language model to generate five different variations of the user query. For every LLM generated query, we then repeat the baseline procedure and pick the unique top 20 documents. For evaluating this procedure we created a larger dataset that consists of 298 reddit submissions and posts in the BestBuyWorkers subreddit  whose entries were labeled by Llama3-70B. We chunked the data by size of 300 and overlap of 50, and using openAI embeddings, we indexed them into ChromaDB. Then, using mixtral-8x7b LLM with a temperature setting of 0 we generated five different queries and retrieved the top 20 unique documents for every original query. For the first 5, 10, 15 and 20 retrievals, we found that the multi-query approach gives F1 scores 0.39 , 0.64, 0.77, 0.73 whereas the baseline approach gives 0.35, 0.605, 0.74, 0.75.

##### Multi-vector Indexing
In this method, we take the given context docs and summarize it using a large language model. We assign a unique id to every summarized document in order to identify it with the original document. The summarized docs are then indexed into the vector store. The user’s query is matched against the summarized documents, the top retrieved documents are then identified with the original document which are finally returned as relevant. In our implementation, we used the mixtral-8x7b LLM to generate document summaries. To evaluate this method we again used the Llama3-70B labeled dataset. For the first 5, 10, 15 and 20 retrievals, we found that the multi-vector indexing approach gives F1 scores 0.46, 0.71, 0.77, 0.76  whereas the baseline approach gives 0.39, 0.60, 0.71, 0.75.

#### Conclusions and Future Directions
In this project we created a procedure for parsing, chunking, and loading reddit data into vector stores. Retrieval on these indexed documents was evaluated for a range of embedding models and retrieval pipelines. Clustering, multi-querying, and multi-vector indexing all showed improvements over the naive process. Clustering and multi-vector require additional pre-processing that should be considered as a trade-off prior to being implemented at a large scale.
There are a number of additional methodologies that could aid in retrieval quality. These methods would be aided by a more extensively labeled evaluation dataset spanning a majority of the subreddit, as well as other subreddits. Future work on this project could investigate improvements by using hypothetical document embeddings to sample a broader range of the embedded space, searching metadata (self-querying) to make use of timestamps and a sentiment metric generated from the statement. Additionally, we could explore making use of metrics based on the frequency and average length of posts by a given author and the length of the thread from which the statement is sourced.

## Code Description
The source code for the project is contained within the [<code>src</code>](https://github.com/peter-mm-williams/aware-nlp/tree/main/src) directory. It consists of 4 sub-packages:
1. [<code>evaluator</code>](https://github.com/peter-mm-williams/aware-nlp/tree/main/src/evaluator): This contains the definitions of classes used to evaluate the quality of derived <code>BaseRetriever</code> classes.
2. [<code>labeler</code>](https://github.com/peter-mm-williams/aware-nlp/tree/main/src/labeler): This contains the definitions of classes used to automate the labeling of statements as relevant to a given set of questions.
3. [<code>retriever</code>](https://github.com/peter-mm-williams/aware-nlp/tree/main/src/retriever): This contains the class definitions of different retriever pipelines.
4. [<code>util</code>](https://github.com/peter-mm-williams/aware-nlp/tree/main/src/util): This contains some utility functions used repeatedly elsewhere in the codebase.

## Install Directions
##### Option A: Create a conda environment
1. Open terminal/powershell
2. Clone the repo
3. Navigate to the directory <code>"path_to_repo"/"repo_name"</code>
4. Run <code>conda env create -f rag.yml</code>

##### Option B: Create a python virtual environment
1. Open terminal/powershell
2. Clone the repo
3. Navigate to the directory <code>"path_to_repo"/"repo_name"</code>
4. Run <code>python -m virtualenv venv</code>
5. Run <code>source myenv/bin/activate</code>
6. Run <code>pip install -r requirements.txt</code>

##### Installing Ollama (Optional)
In order to run large language models (llm) locally, download ollama and use ollama to download the corresponding llm. Note: this is only necessary for a few notebooks/ scripts (in particular PeterNotebooks/ollama-evaluator.ipynb)
1. [Download ollama](https://ollama.com/download)
2. Open terminal/powershell
3. Run <code>ollama pull "model name"</code>, where you replace "model name" with the desired model ("llama2" for example).
For more comprehensive directions click [here](https://python.langchain.com/docs/integrations/llms/ollama/)
