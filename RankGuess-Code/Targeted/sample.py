import torch
import pickle
from tqdm import tqdm
from torch.distributions import Categorical

stoi = pickle.load(open("./data/ClixSense_stoi.pickle", "rb"))
itos = pickle.load(open("./data/ClixSense_itos.pickle", "rb"))

vocab_size = len(stoi)
device = torch.device("cpu")

from model import Guesser

model = Guesser(vocab_size).cpu()
model.load_state_dict(
    torch.load("",
               map_location=torch.device('cpu')))
model.eval()

random_seed = 1234
torch.manual_seed(random_seed)

output_file = "./data/ClixSense_sample.txt"
num_passwords = 10000

with open(output_file, "w") as file:
    probs = []
    end_token = stoi['<end>']
    for _ in tqdm(range(num_passwords), desc="sampling"):
        current_token = stoi['<begin>']
        password = [current_token]
        prob = 1.0
        idx = 0
        try:
            while current_token != end_token:
                input_tensor = torch.tensor(password).unsqueeze(0).cpu()
                output = model(input_tensor)
                distribution = Categorical(output[0, idx, :])

                current_token = distribution.sample().item()
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

print(f"generate {num_passwords} samples and write to file complete.")
