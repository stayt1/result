from datetime import datetime
import torch
import pickle
from tqdm import tqdm
from utils import *

stoi = pickle.load(open("./data/ClixSense_stoi.pickle", "rb"))
itos = pickle.load(open("./data/ClixSense_itos.pickle", "rb"))
vocab_size = len(stoi)
device = torch.device("cpu")
date = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

random_seed = 123
torch.manual_seed(random_seed)

from model import Guesser

model = Guesser(vocab_size).cpu()
model.load_state_dict(
    torch.load("",
               map_location=torch.device('cpu')))
model.eval()

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

def calculate_probability(password):

    try:
        tokens = [stoi[x] for x in password]
    except:
        return 0.0
    tokens.insert(0, stoi['<begin>'])
    input = torch.tensor(tokens).unsqueeze(0).cpu()
    output = model(input)
    prob = 1.0
    for i in range(len(password) - 1):
        prob *= float(output[0, i, tokens[i + 1]])
    prob *= float(output[0, len(password) - 1, stoi['<end>']])
    return prob


input_file = "./data/ClixSense_PI_50_test.csv"
output_file = "./data/ClixSense_pro.txt"

passwords = []

with open(input_file, "r") as file:

    while True:
        try:
            line = file.readline()
        except UnicodeDecodeError:
            continue
        if not line:
            break
        line = line.strip('\r\n')
        line = line.split('\t')
        try:
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
        except IndexError:
            continue
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

        tmp = []
        for items in tagged_passwords:
            tagged_password = [pi_feature[int(str(item)[:2])] if item >= 1000 else chr(item) for item in items]
            tmp.append(tagged_password)
        passwords.append(tmp)

with open(output_file, "w") as file:
    
    for password in tqdm(passwords, desc="progressing"):
        max_prob = 0.0
        pwd = []
        for i in range(len(password)):
            prob = calculate_probability(password[i])
            if prob > max_prob:
                max_prob = prob
                pwd = password[i]
        file.write(f"{''.join(pwd)}\t{max_prob}\n")
file.close()
print("finished!")
