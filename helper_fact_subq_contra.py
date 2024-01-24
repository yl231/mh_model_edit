import torch


from helper_process_dataset import mean_pooling


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


def hard_retrieve_facts(subquestion_idx, id, query_phrase, new_facts, caseid_to_qa_pair, embs, contriever, tokenizer):
    if id not in caseid_to_qa_pair.keys() or subquestion_idx not in caseid_to_qa_pair[id].keys():
        fact_ids = retrieve_facts(query_phrase, embs, contriever, tokenizer)
        fact_sent = new_facts[fact_ids[0]]
        return fact_sent, None
        # return "No relevant fact. Assume no contradiction.", ""
    else:
        res = caseid_to_qa_pair[id][subquestion_idx]
        return res[1], res[2]


def get_next_sub_question(case_id, qid, subquestion, caseid_to_sub_questions):
    if (len(caseid_to_sub_questions[case_id].keys())) <= qid:
        return subquestion[len("Subquestion: "):]
    
    return caseid_to_sub_questions[case_id][qid]