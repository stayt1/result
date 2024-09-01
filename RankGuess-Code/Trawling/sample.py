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

from model import Guesser
model = Guesser(96).cpu()
model.load_state_dict(torch.load("", map_location=torch.device('cpu')))
model.eval()

random_seed = 3407
torch.manual_seed(random_seed)

output_file = ""
num_passwords = 10000

with open(output_file, "w") as file:
    probs = []
    end_token = stoi['<end>']
    for _ in tqdm(range(num_passwords),desc="process"):
        current_token = stoi['<begin>']
        password = [current_token]
        prob = 1.0
        idx = 0
        try:
            while idx <= 18:
                input_tensor = torch.tensor(password).unsqueeze(0).cpu()
                output = model(input_tensor)
                distribution = Categorical(output[0, idx, :])
                current_token = distribution.sample().item()

                if current_token == end_token:
                    break
                else:
                    prob *= float(output[0, idx, current_token])
                    password.append(current_token)
                    idx += 1
        except:
            continue
        try:
            password = [itos[x] for x in password]
        except:
            continue
        password = ''.join(password[1:-1])

        if len(password) < 6:
            continue
        probs.append(prob)
    probs.sort(reverse=True)
    file.write('\n'.join([str(x) for x in probs]))
file.close()

print(f"generate {num_passwords} samples.")
