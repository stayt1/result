import os.path
import random

datasets_list = ['Rockyou', 'Taobao', 'Dodonew', 'Wishbone', '000Webhost', 'LinkedIn']
output_base_path = ''

mixed_data = []

for data_name in datasets_list:
    datafile_path = os.path.join(output_base_path, f'{data_name}/{data_name}_100w.txt')
    with open(datafile_path, 'r') as file:
        lines = file.readlines()

    valid_lines = [line.strip() for line in lines]
    mixed_data += valid_lines

random.shuffle(mixed_data)

output_file = os.path.join(output_base_path, f'mixed_dataset.txt')
with open(output_file, 'w') as file:
    for data in mixed_data:
        file.write(data + '\n')
file.close()
