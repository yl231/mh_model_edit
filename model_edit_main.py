import os
import json
import random
from tqdm import tqdm
import torch

from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
import argparse
import logging
import atexit

from transformers import StoppingCriteria, StoppingCriteriaList

from helper_run_model import remove_extra_target_occurrences, call_model, able_to_quit, ga_contradict_fc2, fetch_rel_subj2subq, \
    break_down_into_subquestions
from helper_process_dataset import get_sent_embeddings, process_datasets, get_ent_rel_id, process_kg, get_subject, \
    get_ent_alias, get_hardcode_rels_breakdown
from helper_fact_subq_contra import retrieve_facts, hard_retrieve_facts, get_next_sub_question
from kg_utils import get_relation, get_fact_form_kg, fit_subject_on_kg

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)


# log_file_path = '/content/drive/MyDrive/MQUAKE_replicate/mh_model_edit_pipelines/outputs/logfile.log'
# file_handler = logging.FileHandler(log_file_path)
# logger.addHandler(file_handler)


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


def save_logger_setup(logger_to_save, file_path, delete_duplicate_output_file):
    """
        existing file_path will result in a new file with an different name to be saved.
    """
    
    # Create a file handler and set the formatter
    if os.path.exists(file_path):
        if delete_duplicate_output_file:
            os.remove(file_path)
        else:
            file_path = file_path.split('.txt')[0] + "(0).txt"
    else:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    file_handler = logging.FileHandler(file_path)
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    
    # Add the file handler to the logger
    logger_to_save.addHandler(file_handler)
    atexit.register(save_log_to_file)


# Function to be called before program exits
def save_log_to_file():
    logger.info("Saving logging information to a file...")


def print_arguments(arguments):
    res = ""
    for key, value in arguments.items():
        res = res + str(key) + ": " + str(value) + "\n"
    return res[:-1]


def main():
    parser = argparse.ArgumentParser(description='command line arguments')
    parser.add_argument('--model_name', type=str, help='Model for the edits.')
    parser.add_argument('--device', type=str, help='Cuda or CPU?')
    parser.add_argument('--file_path', type=str, help='directory path to files')
    parser.add_argument('--seed', type=int, help='random seed number')
    parser.add_argument('--subquestion_breakdown', type=bool, default=False,
                        help='whether to use hard-code subquestion breakdown.')
    parser.add_argument('--fact_retrieve', type=bool, default=False, help='whether to use hard-code fact retrieval.')
    parser.add_argument('--cot_contradiction', type=bool, default=False,
                        help='whether to use cot method in determine presence of '
                             'contradiction.')
    parser.add_argument('--output_dir', type=str, help='output directory')
    parser.add_argument('--delete_duplicate_output_file', type=bool, help='Delete duplicate output file?')
    parser.add_argument('--start', type=int, default=0, help='start pos of dataset')
    parser.add_argument('--end', type=int, default=200, help='end pos of dataset')
    parser.add_argument('--fact_query_on', type=str, default="subquestion", help='')
    parser.add_argument('--edit_num', type=int, default=1000, help='number of questions to edit')
    parser.add_argument('--holistic_cot', type=bool, default=False, help='holistic COT')
    parser.add_argument('--print_prompt', type=bool, default=False, help='print the prompt for debug')
    parser.add_argument('--dataset', type=str, default="CF", help='default counterfactual')
    parser.add_argument('--kg_walk', type=bool, default=False, help="whether to use kg_walk")
    parser.add_argument('--hops', type=int, default=4, help="number of hops")
    parser.add_argument('--breakdown_first', type=bool, default=False, help="Breakdown subq first.")
    parser.add_argument('--postfix_breakdown_prompt', type=str, default='')
    
    # parser.add_argument()
    
    # parse arguments from
    args = parser.parse_args()
    model_name = args.model_name
    device = args.device
    file_path = args.file_path
    seed_num = args.seed
    fact_retrieve = args.fact_retrieve
    subquestion_breakdown = args.subquestion_breakdown
    cot_contradiction = args.cot_contradiction
    output_dir = args.output_dir
    start = args.start
    end = args.end
    fact_query_on = args.fact_query_on
    edit_num = args.edit_num
    holistic_cot = args.holistic_cot
    name_of_the_run = "%s_%s%s%s_%s_%s" % (model_name, str(fact_retrieve)[0],
                                           str(subquestion_breakdown)[0],
                                           str(cot_contradiction)[0],
                                           start, end)
    delete_duplicate_output_file = args.delete_duplicate_output_file
    print_prompt = args.print_prompt
    dataset_name = "-" + args.dataset
    kg_walk = args.kg_walk
    hops = args.hops
    breakdown_first = args.breakdown_first
    postfix_breakdown_prompt = args.postfix_breakdown_prompt
    
    save_logger_setup(logger, output_dir + "%s.txt" % name_of_the_run, delete_duplicate_output_file)
    
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

        # this ends he block:
        sc_end_block = StoppingCriteriaList([StoppingCriteriaSub(stops=[2023, 4515, 1996, 3796])])
    
    else:
        raise ValueError("Model <%s> not implemented yet." % model_name)
    
    if not kg_walk:
        if fact_query_on == "generated_answer_formatted":
            with open(file_path + 'prompts/fill_out_generated_answer.txt', 'r', encoding='utf-8') as f:
                task_prompt = f.read()
        elif holistic_cot and cot_contradiction:
            # with open(file_path + 'prompts/MeLLo-prompt.txt', 'r', encoding='utf-8') as f:
            with open(file_path + 'prompts/Mello_holistic_prompt.txt', 'r', encoding='utf-8') as f:
                task_prompt = f.read()
        else:
            with open(file_path + 'prompts/MeLLo-prompt.txt', 'r', encoding='utf-8') as f:
                # with open(file_path + 'prompts/Mello_holistic_prompt.txt', 'r', encoding='utf-8') as f:
                task_prompt = f.read()
        
        with open(file_path + 'prompts/TaskPromptEdit_hs.txt', 'r', encoding='utf-8') as f:
            cot_prompt3 = f.read()
        
        dataset_path = "MQuAKE-CF-3k"
        if dataset_name == '-T':
            dataset_path = "MQuAKE-T"
        with open(file_path + f'datasets/{dataset_path}.json', 'r') as f:
            dataset = json.load(f)
        
        # caseid_to_qa_pair is for fact-retrieval, caseid_to_sub_questions is for subq breakdown.
        new_facts, caseid_to_qa_pair, caseid_to_sub_questions, rand_list = process_datasets(dataset, file_path,
                                                                                            seed_num=seed_num,
                                                                                            edit_num=edit_num,
                                                                                            dataset_name=dataset_name)
        
        # retrieve model:
        contriever = AutoModel.from_pretrained("facebook/contriever-msmarco").to(device)
        tokenizer = AutoTokenizer.from_pretrained("facebook/contriever-msmarco")
        
        # embedding of facts in retrieve model:
        embs = get_sent_embeddings(new_facts, contriever, tokenizer)
        
        logger.info("Prepare works are Done!")
        
        evaluate_on_dataset_full_functionality(dataset=dataset,
                                               task_prompt=task_prompt,
                                               new_facts=new_facts,
                                               caseid_to_qa_pair=caseid_to_qa_pair,
                                               caseid_to_sub_questions=caseid_to_sub_questions,
                                               embs=embs,
                                               fact_retrieve=fact_retrieve,
                                               subquestion_breakdown=subquestion_breakdown,
                                               cot_contradiction=cot_contradiction,
                                               sc_fact=sc_facts,
                                               sc_subq=sc_subq,
                                               sc_contra=sc_done,
                                               rand_list=rand_list,
                                               model=model,
                                               gptj_tokenizer=gptj_tokenizer,
                                               device=device,
                                               S=start,
                                               T=end,
                                               contriever=contriever,
                                               tokenizer=tokenizer,
                                               cot_prompt=cot_prompt3,
                                               fact_query_on=fact_query_on,
                                               holistic_cot=holistic_cot,
                                               print_prompt=print_prompt,
                                               dataset_name=dataset_name
                                               )
    
    else:  # to use kg walk
        with open(file_path + 'prompts/fill_out_ga_w_blank2.txt', 'r', encoding='utf-8') as f:
            task_prompt = f.read()
        
        if breakdown_first:
            try:
                with open(file_path + f'prompts/subq_breakdown{postfix_breakdown_prompt}.txt', 'r', encoding='utf-8') as f:
                    breakdown_prompt = f.read()
            except Exception:
                raise ValueError(f"postfix_breakdown_prompt = {postfix_breakdown_prompt} not created for breakdown prompt yet.")
            
            with open(file_path + 'prompts/relation2subq_prompt2.txt', 'r', encoding='utf-8') as f:
                relation2subq_prompt = f.read()
        
        if dataset_name == '-T':
            instance_num = 1868  # currently only for -CF.
            with open(file_path + f'datasets/MQuAKE-T.json', 'r') as f:
                dataset = json.load(f)
        elif dataset_name == '-CF':
            instance_num = 3000
            with open(file_path + f'datasets/MQuAKE-CF-3k-idMatched.json', 'r') as f:
                dataset = json.load(f)
        else:
            raise ValueError("Not implemented for dataset %s. " % dataset_name)
        rand_list = random.sample(range(instance_num), edit_num)
        
        entity2id, id2entity, rel2id, id2rel = get_ent_rel_id(file_path, dataset)
        edit_kg, kg_s_r_o = process_kg(dataset, rand_list)
        ent2alias, alias2id = get_ent_alias(dataset, rand_list, entity2id)
        
        # retrieve model:
        contriever = AutoModel.from_pretrained("facebook/contriever-msmarco").to(device)
        tokenizer = AutoTokenizer.from_pretrained("facebook/contriever-msmarco")
        
        rels = list(rel2id.keys())
        rel_emb = get_sent_embeddings(rels, contriever, tokenizer)
        
        ents = list(entity2id.keys())
        ent_emb = get_sent_embeddings(ents, contriever, tokenizer)
        
        logger.info("Prepare works are Done!")
        
        if breakdown_first:
            evaluate_on_dataset_kg_walk_breakdown_first(dataset=dataset,
                                                        task_prompt=task_prompt,
                                                        sc_facts=sc_facts,
                                                        model=model,
                                                        gptj_tokenizer=gptj_tokenizer,
                                                        device=device,
                                                        rels=rels,
                                                        rel_emb=rel_emb,
                                                        contriever=contriever,
                                                        tokenizer=tokenizer,
                                                        subq_breakdown=subquestion_breakdown,
                                                        entity2id=entity2id,
                                                        ent2alias=ent2alias,
                                                        rel2id=rel2id,
                                                        kg_s_r_o=kg_s_r_o,
                                                        id2entity=id2entity,
                                                        ent_emb=ent_emb,
                                                        ents=ents,
                                                        rand_list=rand_list,
                                                        print_prompt=print_prompt,
                                                        breakdown_prompt=breakdown_prompt,
                                                        sc_end_block=sc_end_block,
                                                        relation2subq_prompt=relation2subq_prompt,
                                                        sc_done=sc_done,
                                                        S=start,
                                                        T=end)
        else:
            evaluate_on_dataset_kg_walk(dataset=dataset,
                                        task_prompt=task_prompt,
                                        sc_facts=sc_facts,
                                        model=model,
                                        gptj_tokenizer=gptj_tokenizer,
                                        device=device,
                                        rels=rels,
                                        rel_emb=rel_emb,
                                        contriever=contriever,
                                        tokenizer=tokenizer,
                                        entity2id=entity2id,
                                        ent2alias=ent2alias,
                                        rel2id=rel2id,
                                        kg_s_r_o=kg_s_r_o,
                                        id2entity=id2entity,
                                        ent_emb=ent_emb,
                                        ents=ents,
                                        rand_list=rand_list,
                                        print_prompt=print_prompt,
                                        hops=hops,
                                        S=start,
                                        T=end)
    
    logger.info("Job finished.")


# the full function to complete later:

def evaluate_on_dataset_full_functionality(dataset, task_prompt, new_facts, caseid_to_qa_pair, caseid_to_sub_questions,
                                           embs, fact_retrieve, subquestion_breakdown, cot_contradiction, sc_fact,
                                           sc_subq, sc_contra, rand_list, model, gptj_tokenizer, device, contriever,
                                           tokenizer, cot_prompt, fact_query_on, holistic_cot, print_prompt,
                                           dataset_name,
                                           S=0, T=200):
    # Run MeLLo on the first T (T=200) examples
    
    cor = 0
    tot = 0
    subquestion = 'Subquestion: '
    
    for d in tqdm(dataset[S:T]):
        tot += 1
        for q in d["questions"]:
            found_ans = False
            prompt = task_prompt + "\n\nQuestion: " + q
            num_single_hops = (d['case_id'] - 1) // 1000 + 2
            llm_answer = None
            ans = None
            for i in range(4 + int(subquestion_breakdown)):  # max of 4 hops
                if subquestion_breakdown:
                    if i < num_single_hops:
                        subq = get_next_sub_question(d['case_id'] - 1, i, subquestion, caseid_to_sub_questions)
                        if i != 0:
                            subq = subq.format(llm_answer)
                        prompt = prompt + "\n" + "Subquestion: " + subq + "\nGenerated answer: "
                
                # prompt the model to generate a subquestion and a tentative answer
                
                prompt = call_model(prompt, sc_fact, model, gptj_tokenizer, device)
                if prompt.strip().split('\n')[-1] == 'Retrieved fact:':
                    prompt = prompt[:-len('\nRetrieved fact:')]
                prompt = remove_extra_target_occurrences(prompt, "Question: ", 5)[4:]
                # print(gen[len(task_prompt):])
                # print("-" * 50)
                # if final answer is there, get the answer and exit
                quit, ans = able_to_quit(prompt, task_prompt)
                if quit:
                    found_ans = True
                    break
                
                # otherwise, extract the generated subquestion
                temp_split = prompt.strip().split('\n')
                if len(temp_split) < 2:
                    break  # failed case
                subquestion = temp_split[-2]
                
                if not subquestion.startswith('Subquestion: '):
                    break  # failed case
                
                generated_answer = temp_split[-1]
                query_phrase = subquestion
                if fact_query_on == "generated_answer_formatted":
                    if not temp_split[-1].startswith("Generated answer: "):
                        break
                    # Genertaed answer: XX is {}. YY
                    ga_seg = generated_answer[len("Generated answer: "):].strip().split('. ')
                    
                    if len(ga_seg) == 2:
                        query_phrase = ga_seg[0]
                        answer_object = ga_seg[1]
                
                # True if we want to use hard-code facts retrieval
                if fact_retrieve:
                    fact_sent, _ = hard_retrieve_facts(i, d["case_id"] - 1, query_phrase, new_facts, caseid_to_qa_pair,
                                                       embs, contriever, tokenizer)
                else:
                    fact_ids = retrieve_facts(query_phrase, embs, contriever, tokenizer)
                    fact_sent = new_facts[fact_ids[0]]
                
                # put the retrieved fact at the end of the prompt, the model self-checks if it contradicts
                prompt = prompt + '\nRetrieved fact: ' + fact_sent
                if cot_contradiction and holistic_cot:
                    prompt += "\nNow, let's think step by step, would there be a " \
                              "contradiction in the generated answer to the " \
                              "nearest subquestion given the retrieved fact? "
                
                if i >= num_single_hops:
                    continue
                
                if cot_contradiction:
                    yes_or_no, res = ga_contradict_fc2(subquestion, generated_answer, fact_sent, cot_prompt, sc_contra,
                                                       model, gptj_tokenizer, device, task_prompt)
                    if yes_or_no:
                        do_not = "does"
                    else:
                        do_not = "does not"
                    prompt = prompt + '\n' + "Retrieved fact %s contradict to generated answer, " \
                                             "so the intermediate answer is: %s" % (do_not, res)
                    # print(prompt[len(task_prompt):])
                    # print("-" * 50)ga_contradict_fc2
                    llm_answer = res
                elif subquestion_breakdown:
                    prompt = call_model(prompt, sc_subq, model, gptj_tokenizer, device)[4:]
                    
                    if prompt.strip().split('\n')[-1] == 'Subquestion:':
                        prompt = prompt[:-len('\nSubquestion:')]
                    
                    # print(gen[len(task_prompt):])
                    # print('-' * 100)
                    
                    llm_answer = prompt.strip().split('\n')[-1].split(": ")[-1]
                    # print(last_answer)
                else:
                    continue
                
                quit, ans = able_to_quit(prompt, task_prompt)
                if quit:
                    found_ans = True
                    break
            if print_prompt:
                logger.info(prompt[len(task_prompt):] + "\n")
            if not found_ans:
                continue
            # if the answer is correct
            # print(ans)
            answer = "answer"
            answer_alias = "answer_alias"
            if (d["case_id"] - 1) in rand_list:
                answer = "new_" + answer
                answer_alias = "new_" + answer_alias
            
            # print(d[answer], d[answer_alias])
            if ans == d[answer] or ans in d[answer_alias] or \
                    dataset_name == '-T' and (d["case_id"] - 1) not in rand_list and ans in d["answer_extended"]:
                cor += 1
                break
        logger.info("%s, %s" % (cor, tot))
        # print("-" * 100)
        # print("-" * 100)
    
    logger.info(f'Multi-hop acc = {cor / tot} ({cor} / {tot})')


def evaluate_on_dataset_kg_walk(dataset, task_prompt, sc_facts, model, gptj_tokenizer, device, rels, rel_emb,
                                contriever, tokenizer, entity2id, ent2alias, rel2id, kg_s_r_o, id2entity, ent_emb, ents,
                                rand_list, print_prompt, hops, S=0, T=200):
    cor = 0
    tot = 0
    
    for d in tqdm(dataset[S:T]):
        if print_prompt:
            print("=" * 50, f"Caseid = {d['case_id']}", "=" * 50)
        tot += 1
        for q_id, q in enumerate(d["questions"]):
            if print_prompt:
                print("=" * 30, f"q_id = {q_id + 1}", "=" * 30)
            found_ans = False
            ans = None
            subject = get_subject(d)
            prompt = task_prompt + "\n\nQuestion: " + q
            for i in range(hops):
                prompt = call_model(prompt, sc_facts, model, gptj_tokenizer, device)
                if prompt.strip().split('\n')[-1] == 'Retrieved fact:':
                    prompt = prompt[:-len('\nRetrieved fact:')]
                prompt = remove_extra_target_occurrences(prompt, "Question: ", 5)[4:]
                # print(gen[len(task_prompt):])
                # print("-" * 50)
                
                # if final answer is there, get the answer and exit
                if prompt.count('Final answer: ') >= 5:
                    found_ans = True
                    index = prompt.find('Final answer: ', len(task_prompt))
                    final_ans = prompt[index:]
                    final_ans = final_ans.strip().split('\n')[0][len('Final answer: '):]
                    if final_ans == ans:
                        ans = final_ans
                    break
                
                temp_split = prompt.strip().split('\n')
                # otherwise, extract the generated subquestion
                if len(temp_split) < 2:
                    break  # failed case
                
                subquestion = temp_split[-2]
                if not subquestion.startswith('Subquestion: '):
                    break  # failed case
                subquestion = subquestion[len("Subquestion: "):]
                
                relation = get_relation(subquestion, rels, rel_emb, contriever, tokenizer)
                
                generated_answer = temp_split[-1][len("Generated answer: "):]
                
                # Genertaed answer: XX is {}. YY
                ga_seg = generated_answer.strip().split('. ')
                query_phrase = subquestion
                
                if len(ga_seg) >= 2:
                    query_phrase = ga_seg[0]
                    answer_object = ". ".join(ga_seg[1:])
                else:
                    break
                
                fact_sent, contra_or_not, fact_object = get_fact_form_kg(subject, relation, entity2id, ent2alias,
                                                                         rel2id, kg_s_r_o, id2entity)
                
                # check whether there is a contradiction:
                # contra_promt = "Retrieved fact {} to generated answer, so the intermediate answer is: {}\n"
                contra_promt = "Retrieved fact {} to generated answer, so continue with this subject: {}.\n"
                if contra_or_not:
                    does_or_doesnot = "contradicts"
                    inter_answer = fact_object
                else:
                    does_or_doesnot = "does not contradict"
                    inter_answer = answer_object
                
                contra_promt = contra_promt.format(does_or_doesnot, inter_answer)
                
                # reset pointer and var for the next hop:
                subject = fit_subject_on_kg(inter_answer, entity2id, ent_emb, contriever, tokenizer, ents,
                                            kg_s_r_o, ent2alias)
                ans = subject
                prompt = prompt + '\nRetrieved fact: ' + fact_sent + '.\n' + contra_promt
                
                # if print_prompt:
                #     print("=" * 20, f"hop {i + 1}", "=" * 20)
                #     print(prompt[len(task_prompt) + 2:])
            
            # if not found_ans:
            #     continue
            if print_prompt:
                # print("=" * 20, "End", "=" * 20)
                print(prompt[len(task_prompt) + 2:])
            
            answer = "answer"
            answer_alias = "answer_alias"
            if (d["case_id"] - 1) in rand_list:
                answer = "new_" + answer
                answer_alias = "new_" + answer_alias
            if print_prompt:
                print("=" * 20, "Answers", "=" * 20)
                print(d[answer], d[answer_alias])
            if ans == d[answer] or ans in d[answer_alias]:
                cor += 1
                break
        logger.info("%s, %s" % (cor, tot))
    
    logger.info(f'Multi-hop acc = {cor / tot} ({cor} / {tot})')


def evaluate_on_dataset_kg_walk_breakdown_first(dataset, task_prompt, sc_facts, model, gptj_tokenizer, device, rels,
                                                rel_emb, subq_breakdown,
                                                contriever, tokenizer, entity2id, ent2alias, rel2id, kg_s_r_o,
                                                id2entity, ent_emb, ents, sc_done, sc_end_block, relation2subq_prompt,
                                                rand_list, print_prompt, breakdown_prompt, S=0,
                                                T=200):
    cor = 0
    tot = 0
    
    for d in tqdm(dataset[S:T]):
        if print_prompt:
            print("=" * 50, f"Caseid = {d['case_id']}", "=" * 50)
        tot += 1
        if subq_breakdown:
            start_subject, breakdown_rels_list = get_hardcode_rels_breakdown(d, rand_list)
        else:
            start_subject, breakdown_rels_list = break_down_into_subquestions(d, breakdown_prompt, sc_done, gptj_tokenizer, model)
        for q_id, q in enumerate(d["questions"]):
            if print_prompt:
                print("=" * 30, f"q_id = {q_id + 1}", "=" * 30)
            subject = start_subject
            if subq_breakdown:
                breakdown_rels = breakdown_rels_list
            else:
                breakdown_rels = breakdown_rels_list[q_id]
            found_ans = False
            ans = None
            prompt = task_prompt + "\n\nQuestion: " + q + "\n"
            for i in range(len(breakdown_rels)):
                relation = breakdown_rels[i]
                rel = get_relation(relation, rels, rel_emb, contriever, tokenizer)
                subquestion = fetch_rel_subj2subq(subject, rel, relation2subq_prompt, sc_end_block, model=model,
                                                  gptj_tokenizer=gptj_tokenizer, device=device)
                # subquestion = rel.format(subject)
                prompt = prompt + "Subquestion: " + subquestion + "\n"
                
                prompt = call_model(prompt, sc_facts, model=model, temperature=0.2, device=device,
                                    gptj_tokenizer=gptj_tokenizer)
                if prompt.strip().split('\n')[-1] == 'Retrieved fact:':
                    prompt = prompt[:-len('\nRetrieved fact:')]
                prompt = remove_extra_target_occurrences(prompt, "Question: ", 5)[4:]
                
                temp_split = prompt.strip().split('\n')
                # otherwise, extract the generated subquestion
                if len(temp_split) < 2:
                    break  # failed case
                
                generated_answer = temp_split[-1][len("Generated answer: "):]
                
                # Genertaed answer: XX is {}. YY
                ga_seg = generated_answer.strip().split('. ')
                query_phrase = subquestion
                
                if len(ga_seg) >= 2:
                    query_phrase = ga_seg[0]
                    answer_object = ". ".join(ga_seg[1:])
                else:
                    break
                
                fact_sent, contra_or_not, fact_object = get_fact_form_kg(subject, rel, entity2id, ent2alias,
                                                                         rel2id, kg_s_r_o, id2entity)
                # print(fact_sent)
                
                # check whether there is a contradiction:
                # contra_promt = "Retrieved fact {} to generated answer, so the intermediate answer is: {}\n"
                contra_promt = "Retrieved fact {} to generated answer, so continue with this subject: {}.\n"
                if contra_or_not:
                    does_or_doesnot = "contradicts"
                    inter_answer = fact_object
                else:
                    does_or_doesnot = "does not contradict"
                    inter_answer = answer_object
                
                contra_promt = contra_promt.format(does_or_doesnot, inter_answer)
                
                # reset pointer and var for the next hop:
                subject = fit_subject_on_kg(inter_answer, entity2id, ent_emb, contriever, tokenizer, ents,
                                            kg_s_r_o, ent2alias)
                ans = subject
                prompt = prompt + '\nRetrieved fact: ' + fact_sent + '.\n' + contra_promt
                
                # print("=" * 20, f"hop {i+1}", "=" * 20)
                # print(prompt[len(task_prompt)+2:])
            
            # if not found_ans:
            #     continue
            if print_prompt:
                print("=" * 20, "End", "=" * 20)
                print(prompt[len(task_prompt) + 2:])
            
            answer = "answer"
            answer_alias = "answer_alias"
            if (d["case_id"] - 1) in rand_list:
                answer = "new_" + answer
                answer_alias = "new_" + answer_alias
            if print_prompt:
                print("=" * 20, "Answers", "=" * 20)
                print(d[answer], d[answer_alias])
            if ans == d[answer] or ans in d[answer_alias]:
                cor += 1
                break
        logger.info("%s, %s" % (cor, tot))
    
    logger.info(f'Multi-hop acc = {cor / tot} ({cor} / {tot})')


if __name__ == '__main__':
    main()
    # subq breakdown
    # generate answer
    # retrieve fact
    # contradiction
