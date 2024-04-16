import json
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate

class OllamaLabeler:
    """
    Class to evaluate the relevance of a statement to a given question using the Ollama LLM interface

    Inputs:
        - llm: Ollama object to be used for querying
        - question: question to be evaluated
        - statement: statement to be evaluated
    
    Output:
        - Dictionary containing the following keys: 
          * relevant: boolean indicating whether the statement is relevant to the question 
          * reason: string explaining why and how the statement is or is not relevant
          * output: the raw output from the LLM

    Methods:
    1. _generate_prompt(question, statement)
        - Generates the prompt to be used for querying the LLM
    2. _format_output(output)
        - Formats the output from the LLM into a dictionary
    3. _invoke_llm(query)
        - Invokes the LLM with the query
    4. evaluate_one(question, statement)
        - Evaluates the relevance of a single statement to a question
    """

    def __init__(self, llm: Ollama):
        self.llm = llm
    
    def _generate_prompt(self, question: str, statement: str) -> str:
        return f'''
            <s>[INST] <<SYS>>
            The following statement (delimited by ```) provided below is a response from an employee at the company of interest. 
            The statement should be taken as is. It cannot be used to further a dialogue with the employee. 
            
            Please format the output as a dictionary with the following keys: "relevant", "reason". 
            Relevant should be a boolean value indicating whether the statement is relevant to the question.
            Reason should be a string explaining why and how the statement is or is not relevant.
            <</SYS>>
            Does the statement below help answer the question: {question}
            ```
            Statement: {statement}
            ```
            [/INST]
        '''.replace('            ','')
    
    def _format_output(self, output: str) -> dict:
        # Parse the output from the LLM into a dictionary
        out_dict = {"output": output}
        output = output.replace('\n', '').replace('```','').strip()
        output = output[output.index('{'):output.rindex('}')+1]
        return json.loads(output) | out_dict

    def _invoke_llm(self, query: str) -> str:
        # Invoke the LLM with the query
        return self.llm.invoke(query)

    def evaluate_one(self, question, statement) -> dict:
        # Evaluate the relevance of a single statement to a question
        query = self._generate_prompt(question, statement).replace('\n', '')
        output = self._invoke_llm(query)
        try:
            return self._format_output(output)
        except:
            return {"relevant": None, "reason": "Error parsing output", "output": output}
        


class OllamaLabelerVariablePrompt:

    def __init__(self, llm: Ollama, template: PromptTemplate):
        self.llm = llm
        self.template = template
    
    def _generate_prompt(self, question: str, statement: str) -> str:
        return self.template.format(
            question=question, 
            statement=statement
        ).replace('            ','')
    
    def _format_output(self, output: str) -> dict:
        # Parse the output from the LLM into a dictionary
        out_dict = {"output": output}
        output = output.replace('\n', '').replace('```','').strip()
        output = output[output.index('{'):output.rindex('}')+1]
        return json.loads(output) | out_dict

    def _invoke_llm(self, query: str) -> str:
        # Invoke the LLM with the query
        return self.llm.invoke(query)

    def evaluate_one(self, question, statement) -> dict:
        # Evaluate the relevance of a single statement to a question
        query = self._generate_prompt(question, statement).replace('\n', '')
        output = self._invoke_llm(query)
        try:
            return self._format_output(output)
        except:
            return {"relevant": None, "reason": "Error parsing output", "output": output}