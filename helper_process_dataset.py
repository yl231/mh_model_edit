import random
from tqdm import tqdm
import torch
import json
import os


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


def process_datasets(dataset, file_path, seed_num=100, edit_num=1000, dataset_name='-CF'):
    random.seed(seed_num)
    if dataset_name == '-CF':
        instance_num = 3000
    elif dataset_name == '-T':
        instance_num = 1868
    else:
        raise ValueError(f"Cannot find instance_num for dataset {dataset_name}")
    
    new_facts = set()
    caseid_to_sub_questions = {}
    
    rand_list_path = f"{file_path}/datasets/hard_code_facts/rand_list_{edit_num}_{seed_num}_{dataset_name}.json"
    if os.path.isfile(rand_list_path):
        with open(rand_list_path, 'r', encoding='utf-8') as f:
            rand_list = json.load(f)
    else:
        rand_list = random.sample(range(instance_num), edit_num)
    
    get_caseid2qa_pair = False
    caseid_to_qa_pair = {}
    caseid_to_qa_pair_path = f"{file_path}/datasets/hard_code_facts/caseid_to_qa_pair_{edit_num}_{seed_num}_{dataset_name}.json"
    if os.path.isfile(caseid_to_qa_pair_path):
        with open(caseid_to_qa_pair_path, 'r', encoding='utf-8') as f:
            caseid_to_qa_pair = json.load(f)
    else:
        get_caseid2qa_pair = True
    
    for n in rand_list:
        d = dataset[n]
        if get_caseid2qa_pair:
            caseid_to_qa_pair[n] = {}
        idx = 0
        for r in d["requested_rewrite"]:
            the_fact = f'{r["prompt"].format(r["subject"])} {r["target_new"]["str"]}'
            new_facts.add(the_fact)
            if get_caseid2qa_pair:
                caseid_to_qa_pair[n][idx] = (r["question"], the_fact, r["target_new"]["str"])
            idx += 1
    
    rand_set = set(rand_list)
    for i in range(instance_num):
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
    
    if caseid_to_qa_pair:
        with open(caseid_to_qa_pair_path, 'w') as file:
            json.dump(caseid_to_qa_pair, file, indent=4)
    
    return new_facts, caseid_to_qa_pair, caseid_to_sub_questions, rand_list


def get_ent_rel_id(file_path, dataset):
    
    if len(dataset) == 1868:
        dataset_name = 'T'
    elif len(dataset) == 3000:
        dataset_name = 'CF'
    else:
        raise ValueError("Dataset length is incorrect. Check dataset.")
    with open(f'{file_path}/datasets/{dataset_name}/entity2id.json', 'r') as f:
        entity2id = json.load(f)

    with open(f'{file_path}/datasets/{dataset_name}/id2entity.json', 'r') as f:
        id2entity = json.load(f)

    with open(f'{file_path}/datasets/{dataset_name}/rel2id.json', 'r') as f:
        rel2id = json.load(f)

    with open(f'{file_path}/datasets/{dataset_name}/id2rel.json', 'r') as f:
        id2rel = json.load(f)
    return entity2id, id2entity, rel2id, id2rel


def process_kg(dataset, rand_list):
    edit_kg = {}
    
    kg_s_r_o = {}
    
    for n in rand_list:
        d = dataset[n]
        fact_tuples = d['orig']['edit_triples']
        for index, (fact_tuple, edit) in enumerate(zip(fact_tuples, d["requested_rewrite"])):
            (s, r, o) = fact_tuple
            
            # ordinary kg construction:
            if s in edit_kg.keys():
                if o in edit_kg[s].keys():
                    if r in edit_kg[s][o]:
                        continue
                    else:
                        edit_kg[s][o].add(r)
                else:
                    edit_kg[s][o] = {r}
            else:
                edit_kg[s] = {o: {r}}
            
            # test if there are sro1 and sro2 contradiction:
            if s in kg_s_r_o.keys():
                if r in kg_s_r_o[s].keys():
                    if o not in kg_s_r_o[s][r]:
                        kg_s_r_o[s][r].add(o)
                else:
                    kg_s_r_o[s][r] = {o}
            else:
                kg_s_r_o[s] = {r: {o}}
    
    return edit_kg, kg_s_r_o


def get_subject(d):
    return d['orig']['triples_labeled'][0][0]


def get_ent_alias(dataset, rand_list, entity2id):
    ent2alias = {}
    alias2id = {}
    for idx, d in enumerate(dataset):
        for hop in d['single_hops']:
            answer = hop['answer']
            if answer not in entity2id.keys():
                print(f"ERROR - {answer} not in the entity2id Key Set.")
                break
            answer_alias = hop['answer_alias']
            ent2alias[answer] = set(answer_alias)
            for alias in answer_alias:
                alias2id[alias] = entity2id[answer]
    
    return ent2alias, alias2id


def get_hardcode_rels_breakdown(d, rand_list):
    triples_labeled = "triples_labeled"
    if (d["case_id"] - 1) in rand_list:
        triples_labeled = "new_" + triples_labeled
    return_rels = [triple[1] for triple in d['orig'][triples_labeled]]
    return d['orig']['triples_labeled'][0][0], return_rels
