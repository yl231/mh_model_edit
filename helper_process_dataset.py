import random
from tqdm import tqdm
import torch
import json


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


def process_datasets(dataset, file_path, seed_num=100, edit_num=1000):
    random.seed(seed_num)
    
    new_facts = set()
    caseid_to_sub_questions = {}
    
    caseid_to_qa_pair_path = f"{file_path}/datasets/hard_code_facts/caseid_to_qa_pair_{edit_num}.json"
    with open(caseid_to_qa_pair_path, 'r', encoding='utf-8') as f:
        caseid_to_qa_pair = json.load(f)
        
    rand_list_path = f"{file_path}/datasets/hard_code_facts/rand_list_{edit_num}.json"
    with open(rand_list_path, 'r', encoding='utf-8') as f:
        rand_list = json.load(f)
    for n in rand_list:
        d = dataset[n]
        # caseid_to_qa_pair[n] = {}
        idx = 0
        for r in d["requested_rewrite"]:
            the_fact = f'{r["prompt"].format(r["subject"])} {r["target_new"]["str"]}'
            new_facts.add(the_fact)
            # caseid_to_qa_pair[n][idx] = (r["question"], the_fact, r["target_new"]["str"])
            idx += 1
    
    rand_set = set(rand_list)
    for i in range(3000):
        caseid_to_sub_questions[i] = {}
        single_hops = "single_hops"
        if i in rand_set:
            single_hops = "new_" + single_hops
        
        last_answer = None
        for idx, hop in enumerate(dataset[i][single_hops]):
            if idx != 0:
                hop['question'] = hop['question'].replace(last_answer, "{}")
            caseid_to_sub_questions[i][idx] = hop['question']
            last_answer = hop['answer']
    
    new_facts = list(new_facts)
    
    return new_facts, caseid_to_qa_pair, caseid_to_sub_questions, rand_list
