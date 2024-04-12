# aware-nlp
Erdos Institute Data Science Boot Camp project partnered with Aware. This project involves the investigation and evaluation of different methodologies for ranking prompt responses.

#### Install Directions
##### Option A: Create a conda environment
1. Open terminal/powershell
2. Clone the repo
3. Navigate to the directory <code>"path_to_repo"/<repo></code>
4. Run <code>conda env create -f rag.yml</code>

##### Option B: Create a python virtual environment
1. Open terminal/powershell
2. Clone the <repo>
3. Navigate to the directory <code>"path_to_repo"/<repo></code>
4. Run <code>python -m virtualenv venv</code>
5. Run <code>source myenv/bin/activate</code>
6. Run <code>pip install -r requirements.txt</code>

##### Installing Ollama (Optional)
In order to run large language models (llm) locally, download ollama and use ollama to download the corresponding llm. Note: this is only necessary for a few notebooks/ scripts (in particular PeterNotebooks/ollama-evaluator.ipynb)
1. [Download ollama](https://ollama.com/download)
2. Open terminal/powershell
3. Run <code>ollama pull "model name"</code>, where you replace "model name" with the desired model ("llama2" for example).
For more comprehensive directions click [here](https://python.langchain.com/docs/integrations/llms/ollama/)
