import os
import random

datasets_dict = {
    'LinkedIn': 'path/to/LinkedIn.txt',
    'Rockyou': 'path/to/Rockyou.txt',
    'Dodonew': 'path/to/Dodonew.txt',
    'Taobao': 'path/to/Taobao.txt',
    'Wishbone': 'path/to/Wishbone.txt',
    '000Webhost': 'path/to/000Webhost.txt',
}

output_base_path = 'path/to/output'
if not os.path.exists(output_base_path):
    os.mkdir(output_base_path)

for data_name, data_path in datasets_dict.items():
    print(f"processing dataset {data_name}")

    output_path = os.path.join(output_base_path, data_name)
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    with open(data_path, 'r') as file:
        lines = file.readlines()

    valid_lines = [line.strip() for line in lines if 9 <= len(line.strip()) <= 18]

    select_number = 1000000
    select_string = random.sample(valid_lines, select_number)

    output_file = os.path.join(output_path, f'{data_name}_100w.txt')
    with open(output_file, 'w') as file:
        for string in select_string:
            file.write(string + '\n')
    file.close()
