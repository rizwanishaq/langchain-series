# LangChain Series

The LangChain series is a collection of libraries and tools designed to facilitate natural language processing tasks using custom Hugging Face models. These models are fine-tuned and optimized for specific programming language-related tasks, enabling developers to generate code, answer programming-related questions, and provide language-specific recommendations.

## Setup

To use the LangChain series, you need to install the required dependencies. You can install them using the following command:

```bash
pip install -q -U transformers einops accelerate peft xformers langchain
```

## Usage

First, import the necessary modules:

```python
from transformers import AutoTokenizer, pipeline
import torch
from langchain.llms import HuggingFacePipeline
from langchain import PromptTemplate, LLMChain
```

Next, define the Hugging Face model and tokenizer you want to use:

```python
model = 'tiiuae/falcon-7b-instruct'
tokenizer = AutoTokenizer.from_pretrained(model)
```

Create the pipeline for text generation using Hugging Face's pipeline function:

```python
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
    model_kwargs={"temperature": 0, "max_length": 128}
)
```

Initialize the HuggingFacePipeline:

```python
llm = HuggingFacePipeline(pipeline=pipe)
```

Define the prompt template using the PromptTemplate class:

```python
template = """
You are an expert in {programming_language} programming.
{query}
"""

prompt = PromptTemplate(
    input_variables=["programming_language", "query"], template=template)
```

Create an instance of the LLMChain using the prompt and HuggingFacePipeline:

```python
llm_chain = LLMChain(prompt=prompt, llm=llm)
```

Now you can use the LangChain model to generate code or programming-related text based on a prompt. For example:

```python
programming_language = "JavaScript"
query = "Write a function that determines the Nth Fibonacci number"

response = llm_chain.run(
    {"programming_language": programming_language, "query": query})

print(response)
```

## Contributing

Contributions to the LangChain series are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.

## Acknowledgments

The LangChain series is built on top of the [Hugging Face Transformers library](https://huggingface.co/docs/transformers/index). We would like to thank the Hugging Face team and the open-source community for their valuable contributions.

For more information about the LangChain series and other related projects, please visit the [LangChain](https://python.langchain.com/docs/get_started/introduction.html).

And for this particular series, I am following the [Chris Alexiuk youtube channel](https://www.youtube.com/watch?v=xnZfTuvVVIY&ab_channel=ChrisAlexiuk)

Happy coding!
