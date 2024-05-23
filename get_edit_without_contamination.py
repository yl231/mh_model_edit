def get_edits_without_contamination(dataset, rand_list, problem_case, edit_flag):
    """
    Inputs:
      dataset: the dataset of interest
      rand_list: a list of caseid of edited cases
      problem_case: one multi-hop case that we want to get the set of edits that wouldn't contaminate it
      edit_flag: a boolean (True if problem_case is an edited case, False otherwise).
            Note this affects the correct path of this instance (resulting in different edits that would contaminate it)
    Outputs:
      nl_facts: a list of natural language edits. e.g. "John Milton is a citizen of Spain"
      triple_labeled: a list of edits in triples of text. e.g. "(John Milton, {} is a citizen of, Spain)"
      triple_ids: similar to above but in id form. E.g. "(Q79759, P27, Q29)", where Q79759, P27, Q29 are ids of entity
      case_index: the "caseid-1" (used for list index accessing) of the case that the j-th edit are in.
  
    NOTE: the returned values may contain duplicate edits (since an edit may come from distinct multi-hop cases).

    """
    
    if edit_flag:
        assert problem_case['case_id'] in rand_list
    
    triples_name = 'new_triples' if edit_flag else 'triples'
    correct_path = problem_case['orig'][triples_name]
    
    nl_facts = []  # a list of natural language edits. e.g. "John Milton is a citizen of Spain"
    triple_labeled = []  # a list of edits in triples of text. e.g. "(John Milton, {} is a citizen of, Spain)"
    triple_ids = []  # similar to above but in id form. E.g. "(Q79759, P27, Q29)", where Q79759, P27, Q29 are ids of
    # entity or relation.
    
    case_index = []  # corresponding case index (starts from 0 for list accessing) of the edit
    
    for d in dataset:
        if d['case_id'] not in rand_list:
            continue
        # want to check if d will contaminate problem_case:
        for edit, edit_extra_info in zip(d['orig']['edit_triples'], d['requested_rewrite']):
            contam_flag = False
            if any((edit[0] == p[0] and edit[1] == p[1] and edit[2] != p[2]) for p in correct_path):
                # if the edit is the same subject and relation but different answer to a specific hop -> contamination
                
                if edit_flag and any(
                        (edit[0] == case_edit[0] and edit[1] == case_edit[1] and edit[2] == case_edit[2]) for case_edit
                        in d['orig']['edit_triples']):
                    # if the edit is in the problem_case itself (Update: this will never get invoked.)
                    print("Invoked here..")
                    continue
                contam_flag = True
            
            # add this edit to the edit bank:
            if not contam_flag:
                nl_facts.append(
                    f'{edit_extra_info["prompt"].format(edit_extra_info["subject"])} {edit_extra_info["target_new"]["str"]}')
                triple_labeled.append(tuple(
                    [edit_extra_info['subject'], edit_extra_info['prompt'], edit_extra_info["target_new"]["str"]]))
                triple_ids.append(edit)
                case_index.append(d['case_id'] - 1)
    
    return nl_facts, triple_labeled, triple_ids, case_index
