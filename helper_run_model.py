def call_model(prompt, stop, model, gptj_tokenizer, device, generate_length=150):
    input_ids = gptj_tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    gen_tokens = model.generate(
        input_ids,
        do_sample=True,
        max_length=len(input_ids[0]) + generate_length,
        stopping_criteria=stop,
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

