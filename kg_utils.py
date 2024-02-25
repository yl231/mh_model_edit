from helper_fact_subq_contra import retrieve_facts


def get_relation(subquestion, rels, rel_emb, contriever, tokenizer):
    rel_idx = retrieve_facts(subquestion, rel_emb, contriever, tokenizer)
    rel = rels[rel_idx[0]]
    return rel


def get_fact_form_kg(subject, rel, entity2id, ent2alias, rel2id, kg_s_r_o, id2entity):
    subject_id = None
    if subject in entity2id.keys():
        subject_id = entity2id[subject]
    else:
        for ent in ent2alias.keys():
            if subject in ent2alias[ent]:
                subject_id = entity2id[ent]
                break
    if subject_id is None:
        return "<no fact>", False, None
    
    # rel is retrived using embedding from all rels in the dataset
    rel_id = rel2id[rel]
    
    if subject_id in kg_s_r_o.keys():
        if rel_id in kg_s_r_o[subject_id].keys():
            fact_object = id2entity[list(kg_s_r_o[subject_id][rel_id])[0]]
            fact = f'{rel.format(subject)} {fact_object}'
            return fact, True, fact_object
    
    return "<no fact>", False, None


def fit_subject_on_kg(subject, entity2id, ent_emb, contriever, tokenizer, ents, kg_s_r_o, ent2alias):
    if subject in entity2id.keys():
        return subject
    
    indices = retrieve_facts(subject, ent_emb, contriever, tokenizer, k=10)
    for idx in indices:
        target = ents[idx]
        if target in kg_s_r_o.keys():
            return target
        if target in ent2alias.keys() and subject in ent2alias[target]:
            return target
    return subject
