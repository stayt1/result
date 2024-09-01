import json
import math
import os
from datetime import datetime
import torch
import pickle

from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from utils import PriorityQueue

dataset_name = ''
train_date = ''
epoch_num = 2

stoi = pickle.load(open(f"/{dataset_name}/stoi.pickle", "rb"))
itos = pickle.load(open(f"/{dataset_name}/itos.pickle", "rb"))
vocab_size = len(stoi)
device = torch.device("cpu")
date = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

random_seed = 123
torch.manual_seed(random_seed)

from model import Guesser

model = Guesser(vocab_size).to(device)
model.load_state_dict(
    torch.load(f"save/{dataset_name}/{train_date}/epoch_{epoch_num}_group_num/Guesser.pth",
               map_location=torch.device('cpu')))
model.eval()


class InferDataset(Dataset):
    
    def __init__(self, passwords, indice, nlps):
        self.passwords = passwords
        self.indice = indice
        self.nlps = nlps

    def __len__(self):
        return len(self.passwords)

    def __getitem__(self, idx):
        password = self.passwords[idx]
        nlp = self.nlps[idx]

        input = password[:self.indice]
        input_tensor = torch.tensor(input, dtype=torch.long)

        return input_tensor, nlp


def generate_passwords(template, guess_num=int(1e7)):

    template = ['<begin>'] + template
    mask_count = template.count('<mask>')
    indices = [index for index, value in enumerate(template) if value == '<mask>']
    embedded_template = [stoi[char] for char in template]

    batch_size = 256
    num_workers = 256
    guess_width = math.ceil(guess_num ** (1 / mask_count))

    passwords = [embedded_template]
    nlps = [0]

    for indice in indices:
        infer_dataset = InferDataset(passwords, indice, nlps)
        dataloader = DataLoader(infer_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        passwords = []
        nlps = []

        for batch in dataloader:
            batch_inputs, batch_nlps = batch[0], batch[1]
            with torch.no_grad():
                output = model(batch_inputs.to(device))

            for i, nlp in enumerate(batch_nlps):
                top_values, top_indices = torch.topk(output[i, -1, :], k=guess_width)
                for value, index in zip(top_values, top_indices):
                    new_nlp = nlp - math.log(value.item())
                    new_password = batch_inputs[i].clone().detach().cpu().tolist()
                    passwords.append(new_password + [index.item()] + embedded_template[indice + 1:])
                    nlps.append(new_nlp)
                    
    str_passwords = []
    for password, nlp in zip(passwords, nlps):
        if stoi['<end>'] in password:
            continue
        str_password = ''.join([itos[index] for index in password if index != stoi['<begin>']])
        str_passwords.append((str_password, nlp.item()))

    str_passwords = sorted(str_passwords, key=lambda x: x[1])
    return str_passwords


def crack_dataset(dataset):

    pivot_base_path = ''
    crack_base_path = ''

    pivot_path = os.path.join(pivot_base_path, dataset)
    pivot_files = os.path.join(pivot_path, f'{dataset}.pickle')

    with open(pivot_files, 'rb') as file:
        templates_dict, template2passwords = pickle.load(file)

    for templates_type in templates_dict:
        templates = templates_dict[templates_type]
        crack_path = os.path.join(crack_base_path, dataset, date, templates_type)
        if not os.path.exists(crack_path):
            os.makedirs(crack_path)

        crack_data = {}
        for template in tqdm(templates, desc=f"Cracking {templates_type} templates"):
            template_list = list(template)
            masked_template = ['<mask>' if char == '\t' else char for char in template_list]
            passwords = generate_passwords(masked_template)

            crack_data[template] = {}

            test_passwords_set = set(template2passwords[template])
            for guess_times, (guessed_password, nlp) in enumerate(passwords):
                if guessed_password in test_passwords_set:
                    crack_data[template][guessed_password] = [nlp, guess_times + 1]

            with open(os.path.join(crack_path, f"Rockyou_crack_Clixsense.json"), "w") as json_file:
                json.dump(crack_data, json_file, indent=4)


if __name__ == '__main__':
    crack_dataset_name = 'Clixsense'
    print(f"crack dataset {crack_dataset_name}")
    crack_dataset(crack_dataset_name)
