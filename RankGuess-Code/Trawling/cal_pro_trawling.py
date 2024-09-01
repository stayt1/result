import argparse
import time
import numpy as np
import torch
import pickle
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
from torch.distributions import Categorical

stoi = pickle.load(open("", "rb"))
itos = pickle.load(open("", "rb"))
vocab_size = len(stoi)
device = torch.device("cpu")


random_seed = 3407
torch.manual_seed(random_seed)

from model import Guesser
model = Guesser(96).cpu()
model.load_state_dict(torch.load("", map_location=torch.device('cpu')))
model.eval()


def calculate_probability(password,device='cpu'):
    try:
        tokens = [stoi[x] for x in password]
    except:
        print(f'password {password} is not in the vocabulary')
        return 1e-50
    tokens.insert(0, stoi['<begin>'])
    input = torch.tensor(tokens).unsqueeze(0).to(device)
    output = model(input)
    prob = 1.0
    for i in range(len(password)):
        prob *= float(output[0, i, tokens[i + 1]])
    return max(prob, 1e-50)

input_file = ""
output_file = ""

with open(input_file, "r") as file:
    passwords = file.read().splitlines()

with open(output_file, "w") as file:
    for password in tqdm(passwords,desc="process"):
        prob = calculate_probability(password)
        file.write(f"{password}\t{prob}\n")
file.close()
print("finished!")
