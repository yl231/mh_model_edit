from helper_process_dataset import process_datasets
import json

dataset_path = "MQuAKE-CF-3k"
dataset_name = '-T'
file_path = r'C:/Users/Louie/Desktop/mh_model_edit/'

if dataset_name == '-T':
    dataset_path = "MQuAKE-T"
with open(file_path + f'datasets/{dataset_path}.json', 'r') as f:
    dataset = json.load(f)
    
    
new_facts, caseid_to_qa_pair, caseid_to_sub_questions, rand_list = \
    process_datasets(dataset, file_path, seed_num=100, edit_num=1000, dataset_name=dataset_name)

