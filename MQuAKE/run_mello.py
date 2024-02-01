import os
import json
import random
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModel

# %% md

#### Set up OpenAI API

# %%

from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM


device = "cuda"
gptj_tokenizer = AutoTokenizer.from_pretrained("lmsys/vicuna-7b-v1.3")
model = AutoModelForCausalLM.from_pretrained("lmsys/vicuna-7b-v1.3").to(device)


def call_model(prompt, stop, generate_length=150):
    input_ids = gptj_tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    gen_tokens = model.generate(
        input_ids,
        do_sample=True,
        max_length=len(input_ids[0]) + generate_length,
        stopping_criteria=stop,
    )
    gen_text = gptj_tokenizer.batch_decode(gen_tokens)[0]
    del input_ids, gen_tokens
    return gen_text


def remove_extra_target_occurrences(gen, target, count):
    occurrences = gen.count(target)
    
    if occurrences <= count:
        return gen
    
    index = 0
    for _ in range(count + 1):
        index = gen.find(target, index) + len(target)
    
    # while index < len(gen) and gen[index:].startswith(target):
    #     index += len(target)
    
    return gen[:index - len(target) - 2]


# %% md

#### Functions for retrieval models (Contriever)

# %%

def mean_pooling(token_embeddings, mask):
    token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
    sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
    return sentence_embeddings


def get_sent_embeddings(sents, contriever, tok, BSZ=32):
    all_embs = []
    for i in tqdm(range(0, len(sents), BSZ)):
        sent_batch = sents[i:i + BSZ]
        inputs = tok(sent_batch, padding=True, truncation=True, return_tensors='pt').to("cuda")
        with torch.no_grad():
            outputs = contriever(**inputs)
            embeddings = mean_pooling(outputs[0], inputs['attention_mask'])
        all_embs.append(embeddings.cpu())
    all_embs = torch.vstack(all_embs)
    return all_embs


def retrieve_facts(query, fact_embs, contriever, tok, k=1):
    inputs = tok([query], padding=True, truncation=True, return_tensors='pt').to("cuda")
    with torch.no_grad():
        outputs = contriever(**inputs)
        query_emb = mean_pooling(outputs[0], inputs['attention_mask']).cpu()
    sim = (query_emb @ fact_embs.T)[0]
    knn = sim.topk(k, largest=True)
    return knn.indices


# %% md

#### Load dataset

# %%

with open('datasets/MQuAKE-CF-3k.json', 'r') as f:
    dataset = json.load(f)

# %% md

#### Build a memory index which contains all the edits

# %%

new_facts = set()
for d in dataset:
    for r in d["requested_rewrite"]:
        new_facts.add(f'{r["prompt"].format(r["subject"])} {r["target_new"]["str"]}')
new_facts = list(new_facts)

# %%

contriever = AutoModel.from_pretrained("facebook/contriever-msmarco").cpu()
tokenizer = AutoTokenizer.from_pretrained("facebook/contriever-msmarco")

# %%

embs = get_sent_embeddings(new_facts, contriever, tokenizer)

# %%

# Run test for retrieval index
fact_ids = retrieve_facts("Who is the president of the US?", embs, contriever, tokenizer)
print(new_facts[fact_ids[0]])

# %% md

#### Run MeLLo

# %%

# read prompts
with open('prompts/MeLLo-prompt.txt', 'r') as f:
    task_prompt = f.read()
stop = ["Retrieved fact:"]

# %%

# Run MeLLo on the first T (T=10) examples
T = 10

cor = 0
tot = 0

for d in tqdm(dataset[:10]):
    tot += 1
    for q in d["questions"]:
        found_ans = False
        prompt = task_prompt + "\n\nQustion: " + q
        for i in range(4):
            # prompt the model to generate a subquestion and a tentative answer
            gen = call_gpt(prompt, stop)
            last_sent = gen.strip().split('\n')[-1]
            
            # if final answer is there, get the answer and exit
            if last_sent.startswith('Final answer: '):
                found_ans = True
                ans = last_sent[len("Final answer: "):]
                break
            
            # otherwise, extract the generated subquestion
            if len(gen.strip().split('\n')) < 2:
                break  # failed case
            subquestion = gen.strip().split('\n')[-2]
            if not subquestion.startswith('Subquestion: '):
                break  # failed case
            subquestion = subquestion[len("Subquestion: "):]
            
            # retrieve an edited fact using the generated subquestion
            fact_ids = retrieve_facts(subquestion, embs, contriever, tokenizer)
            fact_sent = new_facts[fact_ids[0]]
            
            # put the retrieved fact at the end of the prompt, the model self-checks if it contradicts
            prompt = prompt + gen + 'Retrieved fact: ' + fact_sent + '.'
        
        prompt = prompt + gen
        
        if not found_ans:
            continue
        # if the answer is correct
        if ans == d["new_answer"] or ans in d["new_answer_alias"]:
            cor += 1
            break

print(f'Multi-hop acc = {cor / tot} ({cor} / {tot})')
