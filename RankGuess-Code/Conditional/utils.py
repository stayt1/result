import random

import torch
from typing import Tuple
import heapq


def sample(model, stoi, max_seq_len, mask, batch_size=1, device='cpu'):
    
    begin_tokens = stoi['<begin>']
    passwords = torch.full((batch_size, max_seq_len), begin_tokens, dtype=torch.long, device=device)

    with torch.no_grad():
        for i in range(max_seq_len - 1):
            input = passwords[:, :i + 1]
            output = model(input)
            distribution = torch.distributions.Categorical(output[:, i, :])
            sample = distribution.sample().unsqueeze(1)

            # If the values in the mask are equal to stoi['<mask>'],
            # then use the sample; otherwise, use the values from the mask.
            mask_value = stoi['<mask>']
            mask_condition = (mask[:, i] == mask_value).unsqueeze(1)
            next_token = torch.where(mask_condition, sample, mask[:, i].unsqueeze(1))

            passwords[:, i + 1] = next_token.squeeze()

    return passwords.tolist()


def calculate_probability(model, password, masked_password, stoi, device='cpu'):

    try:
        tokens = [stoi[x] for x in password]
    except:
        print(password)
        return 0.0
    tokens.insert(0, stoi['<begin>'])
    prob = 1.0

    for i in range(len(masked_password)):
        input = torch.tensor(tokens[:i + 1]).unsqueeze(0).to(device)
        output = model(input)
        if masked_password[i] == '<mask>':
            prob *= float(output[0, i, stoi[password[i]]])

    input = torch.tensor(tokens).unsqueeze(0).to(device)
    output = model(input)
    prob *= float(output[0, len(masked_password), stoi['<end>']])
    return prob


def replace_chars(input_str, probability=0.5):

    char_list = list(input_str)

    total_chars = len(char_list)
    replace_count = int(total_chars * probability)

    replace_indices = random.sample(range(total_chars), replace_count)

    for index in replace_indices:
        char_list[index] = '\t'

    replaced_string = ''.join(char_list)

    return replaced_string


class QueueItem:
    def __init__(self, password: torch.Tensor, nlp: float):
        self.password = password
        self.nlp = nlp

    def __lt__(self, other: 'QueueItem') -> bool:
        return self.nlp < other.nlp

    def get(self) -> Tuple[torch.Tensor, float]:
        return self.password, self.nlp


class PriorityQueue:
    def __init__(self):
        self.heap = []

    def push(self, item: Tuple[torch.Tensor, float]):
        queue_item = QueueItem(*item)
        heapq.heappush(self.heap, queue_item)

    def pop(self):
        if self.heap:
            return heapq.heappop(self.heap)
        else:
            raise IndexError("pop from an empty priority queue")

    def __len__(self):
        return len(self.heap)

    def get_all_items(self):
        items = []
        while len(self.heap) > 0:
            item = self.pop()
            items.append(item.get())
        return items
