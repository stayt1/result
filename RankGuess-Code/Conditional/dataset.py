import os
import pickle
import random
from collections import defaultdict

import torch
import numpy
from torch.utils.data import Dataset, DataLoader
from torchtext.vocab import build_vocab_from_iterator
from utils import replace_chars


class TrainDataset(Dataset):
    def __init__(self, data_path="", max_seq_len=18,
                 top_ratio=0.01, mask_probability=0.5, save_vocab=True) -> None:
        super(TrainDataset, self).__init__()
        self.data = []
        self.data_count = defaultdict(int)
        self.top_password = []

        self.data_len = 0
        self.mask_probability = mask_probability

        self.masked_data = {}
        self.mask_count = defaultdict(int)

        self.data_path = data_path
        self.max_seq_len = max_seq_len
        self.top_ratio = top_ratio

        self.read_data()

        self.vocab = build_vocab_from_iterator(self.get_vocab(), specials=["<begin>", "<end>", "<mask>"])
        self.stoi = self.vocab.get_stoi()
        self.itos = self.vocab.get_itos()

        self.generate_inputs()
        self.generate_top_k()

        if save_vocab:
            self.save_vocab()

    def generate_top_k(self):
        num_top = int(len(self.data) * self.top_ratio)
        self.data_count = sorted(self.data_count.items(), key=lambda x: x[1], reverse=True)
        top_password = []
        for i in range(num_top):
            data_onehot = [self.stoi[chars] for chars in self.data_count[i][0]]
            data_onehot.insert(0, self.stoi["<begin>"])
            while len(data_onehot) < self.max_seq_len:
                data_onehot.append(self.stoi["<end>"])
            top_password.append(data_onehot[:self.max_seq_len])
        self.top_password = torch.from_numpy(numpy.array(top_password, dtype=numpy.int32)).long()

    def generate_inputs(self):
        masked_passwords = []
        passwords = []

        for text in self.data:
            masked_password = text["masked_password"]
            masked_password_onehot = [self.stoi[chars] for chars in masked_password]
            masked_password_onehot.insert(0, self.stoi["<begin>"])
            while len(masked_password_onehot) < self.max_seq_len:
                masked_password_onehot.append(self.stoi["<end>"])
            masked_passwords.append(masked_password_onehot[:self.max_seq_len])

            password = text["guessed_password"]
            password_onehot = [self.stoi[chars] for chars in password]
            password_onehot.insert(0, self.stoi["<begin>"])
            while len(password_onehot) < self.max_seq_len:
                password_onehot.append(self.stoi["<end>"])
            passwords.append(password_onehot[:self.max_seq_len])

        self.masked_passwords = torch.from_numpy(numpy.array(masked_passwords, dtype=numpy.int32)).long()
        self.passwords = torch.from_numpy(numpy.array(passwords, dtype=numpy.int32)).long()

    def read_data(self):
        with open(self.data_path, 'r') as file:
            while True:
                line = file.readline()
                if not line:
                    break
                password = line.strip('\r\n')
                masked_password = replace_chars(password, self.mask_probability)

                password_tuple = tuple([chars for chars in str(password)])
                masked_password_tuple = tuple(
                    [chars if chars != '\t' else '<mask>' for chars in str(masked_password)])

                if masked_password_tuple not in self.masked_data:
                    self.masked_data[masked_password_tuple] = []
                self.masked_data[masked_password_tuple].append(password_tuple)
                self.mask_count[masked_password_tuple] += 1

                self.data.append({
                    "guessed_password": password_tuple,
                    "masked_password": masked_password_tuple
                })
                self.data_count[password_tuple] += 1

        file.close()
        self.data_len = len(self.data)

    def get_vocab(self):
        for text in self.data:
            password = text["guessed_password"]
            yield [chars for chars in password]

    def save_vocab(self):
        dirname, filename = os.path.split(self.data_path)
        itos_path = os.path.join(dirname, "itos.pickle")
        stoi_path = os.path.join(dirname, "stoi.pickle")
        pickle.dump(self.itos, open(itos_path, "wb"))
        pickle.dump(self.stoi, open(stoi_path, "wb"))

    def __getitem__(self, index):
        return self.passwords[index], self.masked_passwords[index], self.top_password[index % len(self.top_password)]

    def __len__(self):
        return self.data_len


class ValDataset(Dataset):
    def __init__(self, data_path="", max_seq_len=18, mask_probability=0.5, val_rate=0.01) -> None:
        super(ValDataset, self).__init__()

        self.top_10 = []

        self.data = []
        self.data_path = data_path
        self.data_len = 0
        self.mask_probability = mask_probability
        self.val_rate = val_rate

        self.data_count = defaultdict(int)

        self.masked_data = {}
        self.mask_count = defaultdict(int)

        self.stoi, self.itos = self.read_vocab()
        self.read_data()
        self.generate_top_10()

    def read_data(self):
        with open(self.data_path, 'r') as file:
            lines = file.readlines()

        lines = [line.strip('\r\n') for line in lines]
        num_lines_to_select = int(len(lines) * self.val_rate)
        val_passwords = random.sample(lines, num_lines_to_select)

        for password in val_passwords:
            masked_password = replace_chars(password, self.mask_probability)

            password_tuple = tuple([chars for chars in str(password)])
            masked_password_tuple = tuple(
                [chars if chars != '\t' else '<mask>' for chars in str(masked_password)])

            if masked_password_tuple not in self.masked_data:
                self.masked_data[masked_password_tuple] = []
            self.masked_data[masked_password_tuple].append(password_tuple)
            self.mask_count[masked_password_tuple] += 1

            self.data.append({
                "guessed_password": password_tuple,
                "masked_password": masked_password_tuple
            })
            self.data_count[password_tuple] += 1
        self.data_len = len(self.data)

    def generate_top_10(self):
        self.mask_count = sorted(self.mask_count.items(), key=lambda x: self.mask_count[x[1]], reverse=True)
        for i in range(10):
            mask = self.mask_count[i][0]
            self.top_10.append({
                "masked_password": mask,
                "guessed_password": self.masked_data[mask]
            })

    def read_vocab(self):
        dirname, filename = os.path.split(self.data_path)
        itos_path = os.path.join(dirname, "itos.pickle")
        stoi_path = os.path.join(dirname, "stoi.pickle")
        itos = pickle.load(open(itos_path, "rb"))
        stoi = pickle.load(open(stoi_path, "rb"))
        return stoi, itos

    def __getitem__(self, index):
        return self.data[index]["guessed_password"], self.data[index]["masked_password"]

    def __len__(self):
        return self.data_len


if __name__ == "__main__":
    train_data_path = ''
    dataset = TrainDataset(data_path=train_data_path)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)
    for i, (passwords, masked_passwords, top_password) in enumerate(dataloader):
        print(passwords.shape, masked_passwords.shape, top_password.shape)
        print(passwords)
        print(masked_passwords)
        print(top_password)

        print('-' * 100)

        pwds = []
        for row in passwords:
            str_row = [dataset.itos[int_val] for int_val in row]
            pwds.append(str_row)
        pwds_str = '\n'.join([' '.join(str_row) for str_row in pwds])
        print(pwds_str)

        print('-' * 100)

        masked_pwds = []
        for row in masked_passwords:
            str_row = [dataset.itos[int_val] for int_val in row]
            masked_pwds.append(str_row)
        masked_pwds_str = '\n'.join([' '.join(str_row) for str_row in masked_pwds])
        print(masked_pwds_str)

        print('-' * 100)

        top_pwds = []
        for row in top_password:
            str_row = [dataset.itos[int_val] for int_val in row]
            top_pwds.append(str_row)
        top_pwds_str = '\n'.join([' '.join(str_row) for str_row in top_pwds])
        print(top_pwds_str)

        print('-' * 100)

        break
    print(dataset.stoi)
    print(dataset.itos)
    print(len(dataset.stoi))

    print('=' * 100)

    val_data_path = ''
    dataset = ValDataset(data_path=val_data_path)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0, collate_fn=lambda batch: batch)
    for i, data in enumerate(dataloader):
        passwords, masked_passwords = data[0]
        print(passwords)
        print(masked_passwords)
        break
