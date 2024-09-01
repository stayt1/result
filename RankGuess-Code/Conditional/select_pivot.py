import copy
import os.path
import pickle

datasets = [
        'LinkedIn'
]

for dataset in datasets:
    
    data_path = f"{dataset}/{dataset}.pickle"
    with open(data_path, 'rb') as file:
        templates_dict, template2passwords = pickle.load(file)

    select_templates_dict = {}
    select_template2passwords = {}
    select_num = 50

    for templates_type in templates_dict:
        templates = list(templates_dict[templates_type])
        select_templates = templates[:select_num]
        select_templates_dict[templates_type] = tuple(select_templates)
        for template in select_templates:
            select_template2passwords[template] = copy.deepcopy(template2passwords[template])

    output_path = f'/pivot/{dataset}'
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    with open(os.path.join(output_path, f'{dataset}.pickle'), 'wb') as file:
        pickle.dump((select_templates_dict, select_template2passwords), file)
