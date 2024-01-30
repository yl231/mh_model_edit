from helper_process_dataset import process_datasets
import os
import random
from tqdm import tqdm
import json
import torch
import re

with open('prompts/MeLLo-prompt.txt', 'r', encoding='utf-8') as f:
    task_prompt = f.read()
# with open('datasets/MQuAKE-CF-3k.json', 'r') as f:
with open('datasets/MQuAKE-T.json', 'r') as f:
    dataset = json.load(f)

from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM

device = "cuda"
gptj_tokenizer = AutoTokenizer.from_pretrained("lmsys/vicuna-7b-v1.3")
model = AutoModelForCausalLM.from_pretrained("lmsys/vicuna-7b-v1.3").to(device)

from transformers import StoppingCriteria, StoppingCriteriaList


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


# retrieve facts:
stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=[8015, 2546, 1490, 2114, 29901])])

# subquestion:
stopping_criteria2 = StoppingCriteriaList([StoppingCriteriaSub(stops=[13, 4035, 12470, 29901])])

# Done.
stopping_criteria3 = StoppingCriteriaList([StoppingCriteriaSub(stops=[25632, 29889], length=2)])


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
        # print(outputs[0])
        # print(outputs[0].shape)
        query_emb = mean_pooling(outputs[0], inputs['attention_mask']).cpu()
        # print('-' * 100)
        # print(query_emb)
    sim = (query_emb @ fact_embs.T)[0]
    knn = sim.topk(k, largest=True)
    
    return knn.indices


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


import random

random.seed(100)

new_facts = set()
# 3000 1868
# rand_list = random.sample(range(1868), 1)
rand_list = [299]
# print(1 in rand_list)


for n in rand_list:
    d = dataset[n]
    for r in d["requested_rewrite"]:
        new_facts.add(f'{r["prompt"].format(r["subject"])} {r["target_new"]["str"]}')
new_facts = list(new_facts)

contriever = AutoModel.from_pretrained("facebook/contriever-msmarco").cuda()
tokenizer = AutoTokenizer.from_pretrained("facebook/contriever-msmarco")

embs = get_sent_embeddings(new_facts, contriever, tokenizer)

# Run MeLLo on the first T (T=10) examples
S = 0
T = 200

cor = 0
tot = 0

for d in tqdm(dataset[S:T]):
    tot += 1
    for q in d["questions"]:
        found_ans = False
        ans = None
        prompt = task_prompt + "\n\nQuestion: " + q
        for i in range(4):
            # prompt the model to generate a subquestion and a tentative answer
            
            gen = call_model(prompt, stopping_criteria)
            if gen.strip().split('\n')[-1] == 'Retrieved fact:':
                gen = gen[:-len('\nRetrieved fact:')]
            gen = remove_extra_target_occurrences(gen, "Question: ", 5)[4:]
            # print(gen[len(task_prompt):])
            # print("-" * 50)
            
            # if final answer is there, get the answer and exit
            if gen.count('Final answer: ') >= 5:
                found_ans = True
                index = gen.find('Final answer: ', len(task_prompt))
                ans = gen[index:]
                ans = ans.strip().split('\n')[0][len('Final answer: '):]
                # print("(%s)" % ans)
                break
            
            temp_split = gen.strip().split('\n')
            # otherwise, extract the generated subquestion
            if len(temp_split) < 2:
                break  # failed case
            
            subquestion = temp_split[-2]
            if not subquestion.startswith('Subquestion: '):
                break  # failed case
            subquestion = subquestion[len("Subquestion: "):]
            
            generated_answer = temp_split[-1][len("Generated answer: "):]
            
            # Genertaed answer: XX is {}. YY
            ga_seg = generated_answer.strip().split('. ')
            query_phrase = subquestion
            
            if len(ga_seg) == 2:
                query_phrase = ga_seg[0]
                answer_object = ga_seg[1]
            
            # retrieve an edited fact using the generated subquestion
            fact_ids = retrieve_facts(query_phrase, embs, contriever, tokenizer)
            fact_sent = new_facts[fact_ids[0]]
            
            # # hard-code the retrieve fact:
            # fact_sent = hard_retrieve_facts(subquestion, d["case_id"] - 1)
            
            # put the retrieved fact at the end of the prompt, the model self-checks if it contradicts
            prompt = gen + '\nRetrieved fact: ' + fact_sent + '.'
            # print(prompt)
            # print('-' * 150)
        
        # prompt = prompt + gen
        
        if not found_ans:
            continue
        
        answer = "answer"
        answer_alias = "answer_alias"
        if (d["case_id"] - 1) in rand_list:
            answer = "new_" + answer
            answer_alias = "new_" + answer_alias
        # print(d[answer], d[answer_alias])

        simple_ground_ans = re.sub(r"[^a-zA-Z ]+", '', d[answer]).lower()
        simple_ans = re.sub(r"[^a-zA-Z ]+", '', ans).lower()
        if simple_ground_ans in simple_ans:
            cor += 1
            break
        else:
            break_flag = False
            for alias in d[answer_alias]:
                simple_alias = re.sub(r"[^a-zA-Z ]+", '', alias).lower()
                if simple_alias in simple_ans:
                    cor += 1
                    break_flag = True
                    break
            if break_flag:
                break
        # if ans == d[answer] or ans in d[answer_alias] or (
        #         (d["case_id"] - 1) not in rand_list and ans in d["answer_extended"]):
        #     cor += 1
        #     break
    print(cor, tot)

print(f'Multi-hop acc = {cor / tot} ({cor} / {tot})')
# record:
# 1000 edits:
# 10/50 + 26/150 = 36/200 = 0.18
