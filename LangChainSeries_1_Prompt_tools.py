from transformers import AutoTokenizer, pipeline

import torch

from langchain.llms import HuggingFacePipeline
from langchain import PromptTemplate, LLMChain


model = 'tiiuae/falcon-7b-instruct'

tokenizer = AutoTokenizer.from_pretrained(model)
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
    model_kwargs={"temperature": 0, "max_length": 128}
)

llm = HuggingFacePipeline(pipeline=pipe)


template = """
You are an expert in {programming_language} programming.
{query}
"""

prompt = PromptTemplate(
    input_variables=["programming_language", "query"], template=template)


# Simple Chain
llm_chain = LLMChain(prompt=prompt, llm=llm)


programming_language = "JavaScript"
query = "Write a function that determines the Nth Fibonacci number"

response = llm_chain.run(
    {"programming_language": programming_language, "query": query})

print(response)
