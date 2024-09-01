import os
from datetime import datetime
import argparse

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import TrainDataset, ValDataset
from train import train_per_epoch, val_per_epoch, warm_up_per_epoch
from model import Guesser, Ranker
from torch.utils.tensorboard import SummaryWriter

learning_rate = 1e-3
epochs = 10
warm_up_epochs = 5
group_num = 9
batch_size = 256
gamma = 0.6
date = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

writer = SummaryWriter()

try:
    assert (batch_size % (group_num - 1) == 0)
except AssertionError:
    print('batch_size % (group_num - 1) != 0')
    raise AssertionError

parser = argparse.ArgumentParser()

parser.add_argument('dataset', type=str, default='Rockyou', nargs='?')
args = parser.parse_args()

project_path = ''

dataset = args.dataset
train_data_path = os.path.join(project_path, f'dataset/{dataset}/{dataset}_100w.txt')
val_data_path = os.path.join(project_path, f'dataset/{dataset}/{dataset}_100w.txt')

train_dataset = TrainDataset(train_data_path)
val_dataset = ValDataset(val_data_path)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=32)
val_dataloader = DataLoader(val_dataset, shuffle=True, num_workers=32, collate_fn=lambda batch: batch)

vocab_size = len(train_dataset.stoi)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

Guesser = Guesser(vocab_size=vocab_size).to(device)
Ranker = Ranker(vocab_size=vocab_size).to(device)

g_optimizer = torch.optim.Adam(Guesser.parameters(), lr=learning_rate)
d_optimizer = torch.optim.Adam(Ranker.parameters(), lr=learning_rate)

for epoch in range(warm_up_epochs):
    warm_up_per_epoch(Guesser, Ranker,
                      d_optimizer, train_dataloader,
                      device, group_num, epoch)

save_path = os.path.join(project_path, f'save/{dataset}/{date}')
if not os.path.exists(save_path):
    os.makedirs(save_path)

log_path = os.path.join(save_path, 'log.txt')
with open(log_path, 'w') as file:
    file.write(
        f"dataset:{dataset}\tdate:{date}\tgamma:{gamma}\tlr:{learning_rate}\t"
        f"group_num:{group_num}\tbatch_size:{batch_size}\twarm_up_epochs:{warm_up_epochs}\t"
        f"epochs:{epochs}\n"
    )
file.close()

for epoch in range(epochs):
    loss = train_per_epoch(Guesser, Ranker,
                           g_optimizer, d_optimizer,
                           train_dataloader, device,
                           group_num, epoch, gamma)
    print(f'epoch: {epoch}, loss: {loss}')
    writer.add_scalar('g_loss', loss[0], epoch)
    writer.add_scalar('r_loss', loss[1], epoch)

    prob, top10_prob, top_prob_dict = val_per_epoch(Guesser, val_dataloader, device, epoch)
    writer.add_scalar('top 1w average log prob', prob, epoch)
    print(f'epoch: {epoch}, top 1w average prob: {prob}')
    for pwd, prob in top_prob_dict.items():
        writer.add_scalar(''.join(pwd), prob, epoch)
    print('====================================================')
    model_path = os.path.join(save_path, f'epoch_{epoch}_group_num_{group_num}')
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    torch.save(Guesser.state_dict(), os.path.join(model_path, f'Guesser.pth'))
    torch.save(Ranker.state_dict(), os.path.join(model_path, f'Ranker.pth'))

    with open(log_path, 'a') as file:
        file.write(
            f'epoch:{epoch}, top_1w_prob: {prob}\n'
        )
    file.close()

writer.close()
