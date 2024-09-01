import copy
from enum import Enum
import torch
from torch import Tensor
import sys
sys.setrecursionlimit(10000)

def sample(model, stoi, max_seq_len, batch_size=1, device='cpu'):
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

class TokenType(Enum):
    DIGIT = 1
    LETTER = 2
    OTHER = 3


def get_type(c):
    if c.isdigit():
        return TokenType.DIGIT
    if c.isalpha():
        return TokenType.LETTER
    return TokenType.OTHER


def name_transform(name):
    returns = []
    if name == '|' or name == '':
        return returns
    name_list = name.split('|')
    name_start = 1000
    returns.append((''.join(name_list), name_start))
    name_start += 1
    acronym = ""
    name_length = 0
    for n in name_list:
        if n == "":
            continue
        name_length += 1
        returns.append((n, name_start))
        acronym += n[0]
        returns.append((n.capitalize(), name_start + 1))
        returns.append((n.upper(), name_start + 2))
        name_start += 3
    returns.append((acronym, 1100))
    returns.append((acronym.upper(), 1101))
    returns.append((acronym.capitalize(), 1102))
    returns.append((acronym[0], 1104))
    returns.append((acronym[0].upper(), 1105))
    if name_length > 1:
        returns.append((acronym[1:], 1103))
    return returns


def birth_transform(birth):
    returns = []
    birth_start = 2000
    if len(birth) != 8:
        return returns
    returns.append((birth, birth_start))  # 19940107
    returns.append((birth[0:4], birth_start + 1))  # 1994
    returns.append((birth[2:4], birth_start + 2))  # 94
    returns.append((birth[4:6], birth_start + 3))  # 01
    returns.append((birth[6:8], birth_start + 4))  # 07
    if birth[4] == '0':
        returns.append((birth[0:4] + birth[5], birth_start + 5))  # 19941
        returns.append((birth[2:4] + birth[5], birth_start + 6))  # 941
        returns.append((birth[5:8], birth_start + 7))  # 107
    if birth[6] == '0':
        returns.append((birth[4:6] + birth[7], birth_start + 8))  # 017
        if birth[4] == '0':
            returns.append((birth[5] + birth[7], birth_start + 9))  # 17
    return returns


def email_transform(email):
    returns = []
    email_start = 3000
    email_letter_start = 3100
    email_digit_start = 3200
    at_pos = email.find('@')
    if at_pos == -1:
        return returns
    returns.append((email, email_start))
    returns.append((email[0:at_pos], email_start + 1))
    email_start += 2

    last_type = get_type(email[0])
    last_pos = 0
    for i in range(1, at_pos):
        current_type = get_type(email[i])
        if last_type != current_type:
            returns.append((email[last_pos:i], email_start))
            email_start += 1
            if last_type == TokenType.DIGIT:
                returns.append((email[last_pos:i], email_digit_start))
                email_digit_start += 1
            elif last_type == TokenType.LETTER:
                returns.append((email[last_pos:i], email_letter_start))
                email_letter_start += 1
            last_pos = i
            last_type = current_type

    if last_pos != 0:
        returns.append((email[last_pos:at_pos], email_start))
        if last_type == TokenType.DIGIT:
            returns.append((email[last_pos:at_pos], email_digit_start))
            email_digit_start += 1
        elif last_type == TokenType.LETTER:
            returns.append((email[last_pos:at_pos], email_letter_start))
            email_letter_start += 1

    period_pos = email[at_pos + 1:].find('.')
    if period_pos != -1:
        returns.append((email[at_pos + 1:at_pos + 1 + period_pos], 3300))
    return returns


def account_transform(account):
    returns = []
    if account == "":
        return returns
    account_start = 4000
    account_letter_start = 4100
    account_digit_start = 4200
    returns.append((account, account_start))
    account_start += 1

    last_type = get_type(account[0])
    last_pos = 0

    for i in range(1, len(account)):
        current_type = get_type(account[i])
        if last_type != current_type:
            returns.append((account[last_pos:i], account_start))
            account_start += 1
            if last_type == TokenType.DIGIT:
                returns.append((account[last_pos:i], account_digit_start))
                account_digit_start += 1
            elif last_type == TokenType.LETTER:
                returns.append((account[last_pos:i], account_letter_start))
                account_letter_start += 1
            last_pos = i
            last_type = current_type

    if last_pos != 0:
        returns.append((account[last_pos:], account_start))
        account_start += 1
        if last_type == TokenType.DIGIT:
            returns.append((account[last_pos:], account_digit_start))
            account_digit_start += 1
        elif last_type == TokenType.LETTER:
            returns.append((account[last_pos:], account_letter_start))
            account_letter_start += 1
    return returns


def phone_transform(phone):
    returns = []
    if phone == '':
        return returns
    phone_start = 5000
    returns.append((phone, phone_start))
    returns.append((phone[-4:], phone_start + 1))
    return returns


def gid_transform(gid):
    returns = []
    if len(gid) != 18:
        return returns
    gid_start = 6000
    returns.append((gid, gid_start))
    returns.append((gid[-4:], gid_start + 1))
    returns.append((gid[:6], gid_start + 2))
    return returns


"""
    Tags the passwords based on specific patterns from transformed information.

    Args:
        passwords (str): The actual passwords, e.g., 'wxy123456'.
        password_list (list): List of integer-encoded characters representing the passwords.
                              Normal characters are encoded using Unicode, and special patterns
                              are encoded with specific numbers, e.g., [119, 120, 121, 49, 50, 51, 52, 53, 54]
                              or [1100, 49, 50, 51, 52, 53, 54] are both encodings for 'wxy123456'.
        transformed_information (list): List of tuples containing encoded personal information and
                                        corresponding tags. e.g., [('', 1000), ('', 1001),
                                        ('', 2000), ('02', 2003), ('03', 2004), ('', 4000)].
        tag2information (dict): A dictionary mapping tag values to personal information.
                                e.g., {1000: '', 1001: ''}
        search_depth (int): The current search depth. Used to avoid infinite loops.

    Returns:
        list: A list of lists containing integer-encoded characters representing tagged passwords.
              Each element in the list corresponds to a tagged passwords list.
              e.g., [[119, 120, 121, 49, 50, 51, 52, 53, 54], [1100, 49, 50, 51, 52, 53, 54]]
"""

def tag_password(password, password_list, transformed_information, tag2information, search_depth = 0):
    tagged_passwords = []
    
    if search_depth > 100:
        return tagged_passwords

    for i in range(len(transformed_information)):
        pattern = transformed_information[i][0]
        tag = transformed_information[i][1]
        pos = password.find(pattern)

        if pos != -1:
            tmp_password_list = copy.deepcopy(password_list)
            pattern_length = len(pattern)
            for j in range(pattern_length):
                tmp_password_list[pos + j] = tag
            tmp_password = password[0: pos] + pattern_length * chr(1) + password[pos + pattern_length:]
            results = tag_password(tmp_password, tmp_password_list, transformed_information[i:], tag2information, search_depth + 1)

            tagged_passwords.extend(results)

    tagged_password = []
    i = 0
    while i < len(password_list):
        tagged_password.append(password_list[i])

        if password_list[i] >= 1000:
            i += len(tag2information[password_list[i]]) - 1
        i += 1
            
    tagged_passwords.append(tagged_password)

    return tagged_passwords
