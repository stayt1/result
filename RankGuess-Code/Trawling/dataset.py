import os
import pickle
from collections import defaultdict

import torch
import numpy
from torch.utils.data import Dataset, DataLoader
from torchtext.vocab import build_vocab_from_iterator


class TrainDataset(Dataset):
    def __init__(self, data_path="", max_seq_len=18, top_ratio=0.01, save_vocab=True) -> None:
        super(TrainDataset, self).__init__()
        self.data = []
        self.data_count = defaultdict(int)
        self.top_password = []
        self.inputs = []
        self.data_len = 0

        self.data_path = data_path
        self.max_seq_len = max_seq_len
        self.top_ratio = top_ratio

        self.read_data()

        self.vocab = build_vocab_from_iterator(self.get_vocab(), specials=["<begin>", "<end>"])
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
            data_onehot = [self.stoi[chars] for chars in str(self.data_count[i][0])]
            data_onehot.insert(0, self.stoi["<begin>"])
            while len(data_onehot) < self.max_seq_len:
                data_onehot.append(self.stoi["<end>"])
            top_password.append(data_onehot[:self.max_seq_len])
        self.top_password = torch.from_numpy(numpy.array(top_password, dtype=numpy.int32)).long()

    def generate_inputs(self):
        inputs = []
        for text in self.data:
            data_onehot = [self.stoi[chars] for chars in str(text)]
            data_onehot.insert(0, self.stoi["<begin>"])
            while len(data_onehot) < self.max_seq_len:
                data_onehot.append(self.stoi["<end>"])
            inputs.append(data_onehot[:self.max_seq_len])
        self.inputs = torch.from_numpy(numpy.array(inputs, dtype=numpy.int32)).long()

    def read_data(self):
        with open(self.data_path, 'r') as f:
            while True:
                line = f.readline()
                if not line:
                    break
                line = line[:-1]
                self.data.append(line)
                self.data_count[line] += 1
        f.close()
        self.data_len = len(self.data)

    def get_vocab(self):
        for text in self.data:
            yield [chars for chars in str(text)]

    def save_vocab(self):
        dirname, filename = os.path.split(self.data_path)
        itos_path = os.path.join(dirname, "Rockyou_itos.pickle")
        stoi_path = os.path.join(dirname, "Rockyou_stoi.pickle")
        pickle.dump(self.itos, open(itos_path, "wb"))
        pickle.dump(self.stoi, open(stoi_path, "wb"))

    def __getitem__(self, index):
        return self.inputs[index], self.top_password[index % len(self.top_password)]

    def __len__(self):
        return self.data_len


class ValDataset(Dataset):
    def __init__(self, data_path="", max_seq_len=18) -> None:
        super(ValDataset, self).__init__()

        self.top_password = [
        ]

        self.data = []
        self.data_path = data_path

        self.stoi, self.itos = self.read_vocab()
        self.read_data()

    def read_data(self):
        with open(self.data_path, 'r') as f:
            while True:
                line = f.readline()
                if not line:
                    break
                line = line[:-1]
                self.data.append(line)
        f.close()

    def read_vocab(self):
        dirname, filename = os.path.split(self.data_path)
        itos_path = os.path.join(dirname, "Rockyou_itos.pickle")
        stoi_path = os.path.join(dirname, "Rockyou_stoi.pickle")
        itos = pickle.load(open(itos_path, "rb"))
        stoi = pickle.load(open(stoi_path, "rb"))
        return stoi, itos

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


if __name__ == "__main__":
    data_path = '../Rockyou1M.txt'
    dataset = TrainDataset(data_path=data_path)
