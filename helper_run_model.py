def call_model(prompt, stop, model, gptj_tokenizer, device, generate_length=150, temperature=1.0):
    input_ids = gptj_tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    gen_tokens = model.generate(
        input_ids,
        do_sample=True,
        max_length=len(input_ids[0]) + generate_length,
        stopping_criteria=stop,
        temperature=temperature
    )
    gen_text = gptj_tokenizer.batch_decode(gen_tokens)[0]
    gen_text = gen_text.replace(gptj_tokenizer.eos_token, '')
    
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


def able_to_quit(gen, task_prompt):
    if gen.count('Final answer: ') >= 5:
        index = gen.find('Final answer: ', len(task_prompt))
        ans = gen[index:]
        ans = ans.strip().split('\n')[0][len('Final answer: '):]
        return True, ans
    else:
        return False, None


def ga_contradict_fc2(subquestion, generated_answer, fact_sent, cot_prompt, sc_contra, model, gptj_tokenizer, device, task_prompt,
                      max_try=5):
    prompt = cot_prompt + "Here we have a question of <%s> and a tentive answer <%s> that is not guaranteed to be " \
                          "correct. Suppose we are now aware of another piece of knowledge <%s>. Let's think step by " \
                          "step, would this new information cause a contridication to the tentative answer and what's "\
                          "the answer to the question? Give the answer in the format of Yes/No, answer \nResponse: " \
             % (subquestion[len("Subquestion: "):], generated_answer[len('Generated answer: '):], fact_sent)

    gen = call_model(prompt, sc_contra, model, gptj_tokenizer, device, generate_length=20)
    gen = remove_extra_target_occurrences(gen, "Response:", 5)

    
    res = gen.split('\n')[-1][:-1].split(', ')
    if len(res) != 2:
        if max_try <= 0:
            return False, "Invalid"
        return ga_contradict_fc2(subquestion, generated_answer, fact_sent, cot_prompt, sc_contra, model, gptj_tokenizer,
                                 device, task_prompt, max_try=max_try-1)
    else:
        if res[0] == "Yes":
            res[0] = True
        else:
            res[0] = False
        return res[0], res[1]


def call_model_batch(prompts, stop, gptj_tokenizer, model, generate_length=150, temperature=1.0):
    # Tokenize the list of prompts
    input_ids = gptj_tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).input_ids.cuda()
    
    # Generate text for the batch of prompts
    gen_tokens = model.generate(
        input_ids,
        do_sample=True,
        max_length=input_ids.shape[1] + generate_length,
        stopping_criteria=stop,
        temperature=temperature
    )
    
    # Decode the generated tokens to text for each prompt in the batch
    gen_texts = gptj_tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)
    
    del input_ids, gen_tokens
    
    return gen_texts


def break_down_into_subquestions(d, breakdown_prompt, sc_done, gptj_tokenizer, model):
    subject = d['orig']['triples_labeled'][0][0]
    retval = [subject]
    
    prompts = []
    for i in range(3):
        prompt = breakdown_prompt + f"Given this problem:\n{d['questions'][i]}\nExtract relations in square parentheses into follows:\n\"{subject}->"
        prompts.append(prompt)
    
    # res = call_model(prompt, stopping_criteria)[4:]
    res = call_model_batch(prompts, sc_done, temperature=0.2, model=model, gptj_tokenizer=gptj_tokenizer)
    for i in range(3):
        # print('-'*100)
        temp = res[i][len(breakdown_prompt):]
        temp = temp.split("\n\n")[0]
        temp = temp.strip().split("\n")[-1]
        rels = extract_entities(temp)
        retval.append(rels)
        # print(temp)
        # print(rels)
    return retval[0], retval[1:]


def extract_entities(input_string):
    segments = input_string.split('->')
    
    # Initialize a list to hold the entities
    entities = []
    
    # Loop through each segment and check if it contains an entity within parentheses
    for segment in segments:
        if '(' in segment and ')' in segment:
            # Extract the entity by removing the parentheses
            entity = segment.strip()[1:-1]
            entities.append(entity)
    
    return entities


def fetch_rel_subj2subq(subject, rel, relation2subq_prompt, sc_end_block, model, gptj_tokenizer):
    prompt = relation2subq_prompt + f"Given this relation: \"{rel}\" and this subject: \"{subject}\",\nThe corresponding question is"
    output = call_model(prompt, sc_end_block, temperature=0.2, generate_length=20, model=model, gptj_tokenizer=gptj_tokenizer)
    output = output.strip().split('\n\n')[5]
    output = output.strip().split('\n')[1]
    output = output.strip().split('\"')[1]
    # print(output)
    
    return output