from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, GPTJForCausalLM
from transformers import StoppingCriteria, StoppingCriteriaList
import os
import json
import random
from tqdm import tqdm
import torch
import argparse
import logging

from helper_run_model import break_down_into_subquestions
from model_edit_main import print_arguments

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)


class StoppingCriteriaSub(StoppingCriteria):
    
    def __init__(self, stops=[], length=5):
        StoppingCriteria.__init__(self),
        self.stops = stops
        self.length = length
    
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, stops=[]):
        exit = True
        for i in range(1, self.length, 1):
            # print(input_ids[0][-i], self.stops[-i])
            if input_ids[0][-i] != self.stops[-i]:
                exit = False
        return exit


parser = argparse.ArgumentParser(description='command line arguments')
parser.add_argument('--model_name', type=str, help='Model for the edits.')
parser.add_argument('--dataset', type=str, default="CF", help='default counterfactual')
parser.add_argument('--file_path', type=str, help='directory path to files')

args = parser.parse_args()
model_name = args.model_name
dataset_name = "-" + args.dataset
file_path = args.file_path

device = 'cuda'

arguments = vars(args)
logger.info("Args are parsed. And as follow: \n %s" % print_arguments(arguments))

if model_name == "vicuna-7b":
    gptj_tokenizer = AutoTokenizer.from_pretrained("lmsys/vicuna-7b-v1.3", padding_side='left')
    model = AutoModelForCausalLM.from_pretrained("lmsys/vicuna-7b-v1.3").to(device)
    
    # llm generation stopping criteria:
    # retrieve facts:
    sc_facts = StoppingCriteriaList([StoppingCriteriaSub(stops=[8015, 2546, 1490, 2114, 29901])])
    
    # subquestion:
    sc_subq = StoppingCriteriaList([StoppingCriteriaSub(stops=[13, 4035, 12470, 29901])])
    
    # Done.
    sc_done = StoppingCriteriaList([StoppingCriteriaSub(stops=[25632, 29889], length=2)])
    
    # this ends he block:
    sc_end_block = StoppingCriteriaList([StoppingCriteriaSub(stops=[2023, 4515, 1996, 3796])])

elif model_name == "llama-7b":
    gptj_tokenizer = AutoTokenizer.from_pretrained("NousResearch/Llama-2-7b-chat-hf", padding_side='left')
    model = AutoModelForCausalLM.from_pretrained("NousResearch/Llama-2-7b-chat-hf").to(device)
    
    # llm generation stopping criteria:
    # retrieve facts:
    sc_facts = StoppingCriteriaList([StoppingCriteriaSub(stops=[8015, 2546, 1490, 2114, 29901])])
    
    # subquestion:
    sc_subq = StoppingCriteriaList([StoppingCriteriaSub(stops=[13, 4035, 12470, 29901])])
    
    # Done.
    sc_done = StoppingCriteriaList([StoppingCriteriaSub(stops=[25632, 29889], length=2)])
    
    # this ends the block:
    sc_end_block = StoppingCriteriaList([StoppingCriteriaSub(stops=[2023, 4515, 1996, 3796])])
else:
    raise ValueError(f'Not implemented yet for model {model_name}')

contriever = AutoModel.from_pretrained("facebook/contriever-msmarco").to(device)
tokenizer = AutoTokenizer.from_pretrained("facebook/contriever-msmarco")

try:
    with open(file_path + f'prompts/subq_breakdown.txt', 'r', encoding='utf-8') as f:
        breakdown_prompt = f.read()
except Exception:
    raise ValueError(f"postfix_breakdown_prompt =  not created for breakdown prompt yet.")

if dataset_name == '-T':
    instance_num = 1868  # currently only for -CF.
    with open(file_path + f'datasets/MQuAKE-T.json', 'r') as f:
        dataset = json.load(f)
elif dataset_name == '-CF':
    instance_num = 3000
    with open(file_path + f'datasets/MQuAKE-CF-3k-idMatched.json', 'r') as f:
        dataset = json.load(f)
elif dataset_name == '-CF-9k':
    instance_num = 9218
    with open(file_path + f'datasets/MQuAKE-CF-9k-idMatched.json', 'r') as f:
        dataset = json.load(f)
else:
    raise ValueError("Not implemented for dataset %s. " % dataset_name)



rels_per_question = []
for idx in range(len(dataset)):
    d = dataset[idx]
    rels_per_question.append(break_down_into_subquestions(d,
                                                          breakdown_prompt,
                                                          sc_done,
                                                          gptj_tokenizer,
                                                          model)[1])
    if not idx % 10:
        with open(file_path + f'datasets/rels_per_question_{dataset_name}_{model_name}_pre-fix.json', 'w') as file:
            json.dump(rels_per_question, file)
        logger.info(f"Saved idx {idx + 1}.")

with open(file_path + f'datasets/rels_per_question_{dataset_name}_{model_name}_pre-fix.json', 'w') as file:
    json.dump(rels_per_question, file)

logger.info(f"All saved.")