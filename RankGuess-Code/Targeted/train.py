import math

import torch
from torch import Tensor
import torch.nn.functional as F

import pickle
import time
from tqdm import tqdm

from utils import sample, calculate_probability

cos_sim = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)


def warm_up_per_epoch(Guesser, Ranker, d_optimizer,
                      data_loader, device, group_num, epoch_num):
    for batch_idx, inputs in tqdm(enumerate(data_loader),
                                  total=len(data_loader),
                                  desc=f'warm up epoch {epoch_num}'):

        real_data, refer_data = inputs
        real_data = real_data.to(device)
        refer_data = refer_data.to(device)

        batch_size, max_seq_len = real_data.shape

        Guesser.eval()
        generate_data = sample(model=Guesser, stoi=data_loader.dataset.stoi, max_seq_len=max_seq_len,
                               batch_size=batch_size, device=device)
        generate_data = torch.tensor(generate_data).long().to(device)

        mixed_data = []
        group_size = batch_size // (group_num - 1)
        for i in range(group_num):
            generate_data_group = generate_data[: i * group_size]
            real_data_group = real_data[i * group_size:]

            mixed_tensor = torch.cat((generate_data_group, real_data_group), dim=0)
            mixed_tensor = mixed_tensor[torch.randperm(batch_size)]

            mixed_data.append(mixed_tensor)

        ## ===========================train Ranker========================================
        Ranker.train()

        predicts = []
        refer_feature = Ranker(refer_data)
        for i in range(group_num):
            mixed_feature = Ranker(mixed_data[i])
            predicts.append(cos_sim(refer_feature, mixed_feature))
        predicts = torch.stack(predicts, dim=1)
        predicts = F.softmax(predicts, dim=-1)

        targets = torch.linspace(1, 0, group_num).to(device)
        targets = targets.unsqueeze(0).expand(batch_size, -1)
        targets = F.softmax(targets, dim=-1)

        d_loss = F.kl_div(predicts, targets, reduction='batchmean')

        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()


def train_per_epoch(Guesser, Ranker,
                    g_optimizer, d_optimizer,
                    data_loader, device, group_num, epoch_num):
    g_loss_total = 0.0
    d_loss_total = 0.0
    for batch_idx, inputs in tqdm(enumerate(data_loader),
                                  total=len(data_loader),
                                  desc=f'training epoch {epoch_num}'):

        real_data, refer_data = inputs
        real_data = real_data.to(device)
        refer_data = refer_data.to(device)

        batch_size, max_seq_len = real_data.shape

        Guesser.eval()
        generate_data = sample(model=Guesser, stoi=data_loader.dataset.stoi, max_seq_len=max_seq_len,
                               batch_size=batch_size, device=device)
        generate_data = torch.tensor(generate_data).long().to(device)

        mixed_data = []
        group_size = batch_size // (group_num - 1)
        for i in range(group_num):
            generate_data_group = generate_data[: i * group_size]
            real_data_group = real_data[i * group_size:]

            mixed_tensor = torch.cat((generate_data_group, real_data_group), dim=0)
            mixed_tensor = mixed_tensor[torch.randperm(batch_size)]

            mixed_data.append(mixed_tensor)

        ## ===========================train Ranker========================================
        Ranker.train()

        predicts = []
        refer_feature = Ranker(refer_data)
        for i in range(group_num):
            mixed_feature = Ranker(mixed_data[i])
            predicts.append(cos_sim(refer_feature, mixed_feature))
        predicts = torch.stack(predicts, dim=1)
        predicts = F.softmax(predicts, dim=-1)

        targets = torch.linspace(1, 0, group_num).to(device)
        targets = targets.unsqueeze(0).expand(batch_size, -1)
        targets = F.softmax(targets, dim=-1)

        d_loss = F.kl_div(predicts, targets, reduction='batchmean')
        d_loss_total += d_loss.item()

        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()

        ## ===========================train Guesser========================================

        Guesser.train()
        gen_feature = Ranker(generate_data)
        refer_feature = Ranker(refer_data)
        rewards = (1 - cos_sim(gen_feature, refer_feature)) / 2

        similarity = F.cosine_similarity(gen_feature.unsqueeze(1), gen_feature.unsqueeze(0), dim=2)
        mask = torch.eye(similarity.size(0), dtype=torch.bool, device=similarity.device)
        similarity = similarity.masked_fill(mask, 0)
        dissimilarity = (1 - similarity) / 2
        penalty = dissimilarity.sum(dim=1) / (dissimilarity.size(1) - 1)

        gamma = 0.3

        rewards = (1 - gamma) * rewards - gamma * penalty

        outputs = Guesser(refer_data)

        outputs_subset = outputs[:, :max_seq_len - 1]
        refer_data_subset = refer_data[:, 1:max_seq_len]
        gathered_probs = outputs_subset.gather(2, refer_data_subset.unsqueeze(-1)).squeeze(-1)
        loss = -torch.log(gathered_probs) * rewards.unsqueeze(1).expand(-1, max_seq_len - 1)
        g_loss = torch.sum(loss)

        last_output = outputs[:, max_seq_len - 1]
        end_token_index = data_loader.dataset.stoi['<end>']
        last_loss = -torch.log(
            last_output.gather(1, torch.tensor([[end_token_index]] * batch_size).to(device))).squeeze(-1) * rewards
        g_loss += torch.sum(last_loss)

        g_loss_total += g_loss.item()

        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()

    return g_loss_total / len(data_loader), d_loss_total / len(data_loader)


def val_per_epoch(Guesser, data_loader, device, epoch_num):
    prob_total = 0.0
    Guesser.eval()
    for batch_idx, inputs in tqdm(enumerate(data_loader), total=len(data_loader), desc=f'validating epoch {epoch_num}'):
        prob = calculate_probability(model=Guesser, password=inputs[0], stoi=data_loader.dataset.stoi, device=device)
        try:
            prob_total += -math.log(prob, 10.0)
        except:
            print(prob)
            exit(1)

    top_prob_dict = {}
    top_prob = 0.0
    for pwd in data_loader.dataset.top_10:
        log_prob = -math.log(
            calculate_probability(model=Guesser, password=pwd, stoi=data_loader.dataset.stoi, device=device), 10.0)
        top_prob_dict[pwd] = log_prob
        top_prob += log_prob
    return prob_total / len(data_loader), top_prob / 10, top_prob_dict


if __name__ == '__main__':
    from model import Guesser, Ranker
    from Targeted.dataset_pii import TrainDataset
    from torch.utils.data import DataLoader

    epochs = 1
    group_num = 9
    batch_size = 128

    train_dataset = TrainDataset()
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    vocab_size = len(train_dataset.stoi)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    Guesser = Guesser(vocab_size=vocab_size).to(device)
    Ranker = Ranker(vocab_size=vocab_size).to(device)

    g_optimizer = torch.optim.Adam(Guesser.parameters(), lr=1e-3)
    d_optimizer = torch.optim.Adam(Ranker.parameters(), lr=1e-3)

    for epoch in range(epochs):
        loss = train_per_epoch(Guesser, Ranker,
                               g_optimizer, d_optimizer,
                               train_dataloader, device,
                               group_num, epoch)
        print(f'epoch: {epoch}, loss: {loss}')