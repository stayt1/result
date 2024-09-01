import copy
import os
import pickle
from collections import defaultdict

import numpy
from torch.utils.data import Dataset, DataLoader
from torchtext.vocab import build_vocab_from_iterator
from utils import *

pi_feature = {
            10: '<FULL_NAME>',
            11: '<NAME_ABBREVIATION>',
            20: '<BIRTH>',
            30: '<EMAIL>',
            31: '<EMAIL_LETTER>',
            32: '<EMAIL_DIGIT>',
            33: '<EMAIL_ADDRESS>',
            40: '<ACCOUNT>',
            41: '<ACCOUNT_LETTER>',
            42: '<ACCOUNT_DIGIT>',
            50: '<PHONE>',
            60: '<ID>'
        }

class TrainDataset(Dataset):
    def __init__(self, data_path="", max_seq_len=18, max_pi_len=86,
                 top_ratio=0.1, save_vocab=True) -> None:
        super(TrainDataset, self).__init__()
        self.data = []
        self.data_count = defaultdict(int)
        self.top_passwords = []
        self.passwords = []
        self.data_len = 0

        self.data_path = data_path
        self.max_seq_len = max_seq_len
        self.max_pi_len = max_pi_len
        self.top_ratio = top_ratio

        self.read_data()

        self.vocab = build_vocab_from_iterator(self.get_vocab(), specials=["<begin>", "<end>"])
        self.stoi = self.vocab.get_stoi()
        self.itos = self.vocab.get_itos()

        self.generate_password()
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
        self.top_passwords = torch.from_numpy(numpy.array(top_password, dtype=numpy.int32)).long()

    def generate_password(self):
        pwd_list = []
        for item in self.data:
            pwd = item['password_list']
            pwd_onehot = [self.stoi[chars] for chars in pwd]
            pwd_onehot.insert(0, self.stoi["<begin>"])
            while len(pwd_onehot) < self.max_seq_len:
                pwd_onehot.append(self.stoi["<end>"])
            pwd_list.append(pwd_onehot[:self.max_seq_len])
        self.passwords = torch.from_numpy(numpy.array(pwd_list, dtype=numpy.int32)).long()

    def read_data(self):
        with open(self.data_path, 'r') as f:
            while True:
                line = f.readline()
                if not line:
                    break
                line = line.strip('\r\n')
                line = line.split('\t')

                data = {
                    'email': line[0],
                    'passwords': line[1],
                    'name': line[2],
                    'gid': line[3],
                    'account': line[4],
                    'phone': line[5],
                    'birth': line[6],
                    'password_list': [ord(char) for char in line[1]]
                }

                '''
                    transformed information is a list of tuples, each tuple contains the transformed information and its tag
                    eg. [('N1', 10), ('N2', 11), ('N3', 20)...]
                '''
                transformed_information = []
                transformed_information.extend(name_transform(data['name']))
                transformed_information.extend(birth_transform(data['birth']))
                transformed_information.extend(email_transform(data['email']))
                transformed_information.extend(account_transform(data['account']))
                transformed_information.extend(phone_transform(data['phone']))
                transformed_information.extend(gid_transform(data['gid']))

                tag2information = {tag: information for information, tag in transformed_information}

                password = data['passwords']
                password_list = data['password_list']

                try:
                    tagged_passwords = tag_password(password, password_list, transformed_information, tag2information)
                except:
                    print(data)
                    exit(0)
                
                for items in tagged_passwords:
                    tagged_password = [pi_feature[int(str(item)[:2])] if item >= 1000 else chr(item) for item in items]
                    self.data.append({
                        'email': line[0],
                        'passwords': line[1],
                        'name': line[2],
                        'gid': line[3],
                        'account': line[4],
                        'phone': line[5],
                        'birth': line[6],
                        'password_list': tagged_password
                    })
                    self.data_count[tuple(tagged_password)] += 1
        f.close()

        self.data_len = len(self.data)

    def get_vocab(self):
        for tagged_password in self.data_count.keys():
            yield tagged_password

    def save_vocab(self):
        dirname, filename = os.path.split(self.data_path)
        itos_path = os.path.join(dirname, "itos.pickle")
        stoi_path = os.path.join(dirname, "stoi.pickle")
        pickle.dump(self.itos, open(itos_path, "wb"))
        pickle.dump(self.stoi, open(stoi_path, "wb"))

    def __getitem__(self, index):
        return self.passwords[index], \
            self.top_passwords[index % len(self.top_passwords)]

    def __len__(self):
        return self.data_len


class ValDataset(Dataset):
    def __init__(self, data_path="", max_seq_len=18, max_pi_len=86,) -> None:
        super(ValDataset, self).__init__()

        self.top_10 = []
        self.max_pi_len = max_pi_len
        self.max_seq_len = max_seq_len
        self.data = []
        self.data_count = defaultdict(int)
        self.data_path = data_path
        self.passwords = []

        self.stoi, self.itos = self.read_vocab()
        self.read_data()
        self.generate_top10()
        self.generate_password()

    def generate_top10(self):
        self.data_count = sorted(self.data_count.items(), key=lambda x: x[1], reverse=True)
        self.top_10 = [item[0] for item in self.data_count[:10]]

    def generate_password(self):
        for item in self.data:
            pwd = item['password_list']
            pwd.insert(0, "<begin>")
            while len(pwd) < self.max_seq_len:
                pwd.append("<end>")
            self.passwords.append(pwd[:self.max_seq_len])

    def read_data(self):
        with open(self.data_path, 'r') as f:
            while True:
                line = f.readline()
                if not line:
                    break
                line = line.strip('\r\n')
                line = line.split('\t')

                data = {
                    'email': line[0],
                    'passwords': line[1],
                    'name': line[2],
                    'gid': line[3],
                    'account': line[4],
                    'phone': line[5],
                    'birth': line[6],
                    'password_list': [ord(char) for char in line[1]]
                }

                '''
                    transformed information is a list of tuples, each tuple contains the transformed information and its tag
                    eg. [('N1', 10), ('N2', 11), ('N3', 20)...]
                '''
                transformed_information = []
                transformed_information.extend(name_transform(data['name']))
                transformed_information.extend(birth_transform(data['birth']))
                transformed_information.extend(email_transform(data['email']))
                transformed_information.extend(account_transform(data['account']))
                transformed_information.extend(phone_transform(data['phone']))
                transformed_information.extend(gid_transform(data['gid']))

                tag2information = {tag: information for information, tag in transformed_information}

                password = data['passwords']
                password_list = data['password_list']

                tagged_passwords = tag_password(password, password_list, transformed_information, tag2information)

                for items in tagged_passwords:
                    tagged_password = [pi_feature[int(str(item)[:2])] if item >= 1000 else chr(item) for item in items]
                    self.data.append({
                        'email': line[0],
                        'passwords': line[1],
                        'name': line[2],
                        'gid': line[3],
                        'account': line[4],
                        'phone': line[5],
                        'birth': line[6],
                        'password_list': tagged_password
                    })
                    self.data_count[tuple(tagged_password)] += 1
        f.close()

        self.data_len = len(self.data)

    def read_vocab(self):
        dirname, filename = os.path.split(self.data_path)
        itos_path = os.path.join(dirname, "")
        stoi_path = os.path.join(dirname, "")
        itos = pickle.load(open(itos_path, "rb"))
        stoi = pickle.load(open(stoi_path, "rb"))
        return stoi, itos

    def __getitem__(self, index):
        return self.passwords[index]

    def __len__(self):
        return len(self.data)


if __name__ == "__main__":
    data_path = 'TrainDataset'
    dataset = TrainDataset(data_path=data_path)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0, drop_last=True)
    for i, (pwd, refer_pwd) in enumerate(dataloader):
        print(pwd.shape, refer_pwd.shape)
        print(pwd)
        print(refer_pwd)
        break
    print(dataset.stoi)
    print(dataset.itos)
    print(len(dataset.stoi))

    data_path = 'data/rootkit/rootkit_PI_addusr_gid_val.txt'
    dataset = ValDataset(data_path=data_path)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0, drop_last=True, collate_fn=lambda batch: batch)
    for i, pwd in enumerate(dataloader):
        print(pwd)
        break
    print(dataset.top_10)
