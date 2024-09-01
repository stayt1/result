import torch
from torch import Tensor

def sample(model, stoi, min_seq_len, max_seq_len, batch_size=1, device='cpu'):
    begin_tokens = stoi['<begin>']
    passwords = Tensor([[begin_tokens]*max_seq_len]*batch_size).long().to(device)
    with torch.no_grad():
        for i in range(max_seq_len - 1):
            input = passwords[:, :i+1]
            output = model(input)
            distribution = torch.distributions.Categorical(output[:, i, :])
            sample = distribution.sample().unsqueeze(0).long().to(device)
            passwords[:, i+1] = sample
    return passwords.tolist()

def calculate_probability(model, password, stoi, device='cpu'):
    try:
        tokens = [stoi[x] for x in password]
    except:
        return 0.0
    tokens.insert(0, stoi['<begin>'])
    input = torch.tensor(tokens).unsqueeze(0).to(device)
    output = model(input)
    prob = 1.0
    for i in range(len(password)-1):
        prob *= float(output[0, i, tokens[i+1]])
    prob *= float(output[0, len(password)-1, stoi['<end>']])
    return prob