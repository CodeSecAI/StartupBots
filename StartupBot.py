from langchain.prompts.prompt import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains.llm_bash.prompt import BashOutputParser
from langchain.chains import LLMBashChain
from langchain.chat_models import AzureChatOpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import LLMChain, ConstitutionalChain
from langchain.chat_models import ChatOpenAI
from langchain.chains.constitutional_ai.models                 import ConstitutionalPrinciple
from langchain.chains import SimpleSequentialChain
from BugDB import *
from KeyStore import *
import json
import os
import argparse
import configparser
parser = argparse.ArgumentParser(
                    prog='CodeSec.ai Model',
                    description='This trains and tests LLM',
                    epilog='CodeSec.ai: The future of code')
parser.add_argument('-m', '--model',required=True)
parser.add_argument('-A','--azure-model',required=True)
parser.add_argument('-t', '--template',required=True)
parser.add_argument('-u','--update',action='store_true')
parser.add_argument('--output',required=False)
args = parser.parse_args()

config = configparser.ConfigParser()
config.read('config.ini')

openai_api_key = config['API_KEYS']['openai_api_key']
azure_api_key = config['API_KEYS']['azure_api_key']
endpoint = config['ENDPOINT']['azure']

def load_jsonl(input_path) -> list:
    """
    Read list of objects from a JSON lines file.
    """
    data = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.rstrip('\n|\r')))
    print('Loaded {} records from {}'.format(len(data), input_path))
    return data

def dump_jsonl(data, output_path, append=True):
    """
    Write list of objects to a JSON lines file.
    """
    mode = 'a+' if append else 'w'
    with open(output_path, mode, encoding='utf-8') as f:
        for line in data:
            json_record = json.dumps(line, ensure_ascii=False)
            f.write(json_record + '\n')

data = load_jsonl(args.model)
template = open(args.template).read()
full_prompt = PromptTemplate.from_template(template)
memory = ConversationBufferMemory()
for i in data:
    memory.chat_memory.add_user_message(i['prompt'])
    memory.chat_memory.add_ai_message(i['completion'])
### Pretrain complete
memory.load_memory_variables({})
llm = AzureChatOpenAI(openai_api_type = "azure",
                          model_name="gpt-35-turbo",
                          deployment_name=args.azure_model,
                          openai_api_base=endpoint,
                          openai_api_version="2023-05-15",
                          openai_api_key=azure_api_key,
                          temperature=0)
while True:
    try:
        question = input(">>> ")
        chain = LLMChain(llm=llm, prompt=full_prompt, memory=memory)
        output = chain.run(question=question)
        print(output)
        memory.chat_memory.add_user_message(question)
        memory.chat_memory.add_ai_message(output)
    except KeyboardInterrupt:
        exit()

    