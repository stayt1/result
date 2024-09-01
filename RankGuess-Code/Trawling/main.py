import os
from datetime import datetime

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import TrainDataset, ValDataset
from train import train_per_epoch, val_per_epoch, warm_up_per_epoch
from Passrank_montecarlo.montecarlo.model import Guesser, Ranker
from torch.utils.tensorboard import SummaryWriter

epochs = 100
warm_up_epochs = 5
group_num = 9
batch_size = 256
date = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
data = 'Rockyou_3_0.6'
writer = SummaryWriter()

try:
    assert (batch_size % (group_num - 1) == 0)
except AssertionError:
    print('batch_size % (group_num - 1) != 0')
    raise AssertionError

train_dataset = TrainDataset()
val_dataset = ValDataset()
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
val_dataloader = DataLoader(val_dataset, shuffle=True, num_workers=0)

vocab_size = len(train_dataset.stoi)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

Guesser = Guesser(vocab_size=vocab_size).to(device)
Ranker = Ranker(vocab_size=vocab_size).to(device)

g_optimizer = torch.optim.Adam(Guesser.parameters(), lr=1e-4)
d_optimizer = torch.optim.Adam(Ranker.parameters(), lr=1e-4)

for epoch in range(warm_up_epochs):
    warm_up_per_epoch(Guesser, Ranker,
                      d_optimizer, train_dataloader,
                      device, group_num, epoch)

for epoch in range(epochs):
    loss = train_per_epoch(Guesser, Ranker,
                           g_optimizer, d_optimizer,
                           train_dataloader, device,
                           group_num, epoch)
    print(f'epoch: {epoch}, loss: {loss}')
    writer.add_scalar('g_loss', loss[0], epoch)
    writer.add_scalar('r_loss', loss[1], epoch)

    prob, top10_prob, top_prob_dict = val_per_epoch(Guesser, val_dataloader, device, epoch)
    writer.add_scalar('top 1w average log prob', prob, epoch)
    writer.add_scalar('top 10 average log prob', top10_prob, epoch)
    print(f'epoch: {epoch}, top 1w average prob: {prob}')
    for pwd, prob in top_prob_dict.items():
        writer.add_scalar(pwd, prob, epoch)
    print('====================================================')
    save_path = f'./save_model/{data}/epoch_{epoch}_group_num_{group_num}'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    torch.save(Guesser.state_dict(), os.path.join(save_path, f'Guesser.pth'))
    torch.save(Ranker.state_dict(), os.path.join(save_path, f'Ranker.pth'))
writer.close()




