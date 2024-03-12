# local-llm
Implementation of local open source LLMs using Python

### ollama-rag/

Contains code from the rag example in the ollama git repository.

### my_ollama-rag.py

My version of the code, using PyPdf to read the pdf file, and changing the prompt. The pdf i'm reading is the one on this git: `jpr-16-2883.pdf`.

## How to run the code

You can install the requirements using the `requirements.txt` file inside **ollama-rag** folder. Install with `pip install -r requirements.txt`. I recommend creating a conda enviroment, or a Python virtual enviroments, to avoid package conflicts.

After that, install Ollama with

`curl -fsSL https://ollama.com/install.sh | sh`

And then pull the llama2 model, and run it from the terminal using 

`ollama pull llama2`

`ollama run llama2`

**This is a very rough prototype, as is this README. I plan on improving this code by a lot. Some improvements i'm planning**
 
- How he reads the PDF, how can i extract maybe specific info from it. 
- How to use more than one PDF on RAG
- How to use chain prompting
- Test different models

