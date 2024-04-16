# aware-nlp
Erdos Institute Data Science Boot Camp project partnered with Aware. This project involves the investigation and evaluation of different methodologies for retrieval for use in RAG (Retrieval-Augmented Generation) systems. In particular, this project investigates retrieval quality for information downloaded from employee subreddits.

## Project Description
#### Constructing the Evaluation Dataset
The Bestbuy subreddit was selected as the focus for quality evaluation.
#### Automated Labeling
#### Quality Variation from Embedding Model Choice

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
