import argparse
import time
import pickle
from torch.utils.data import RandomSampler
from utils import *
import os
import queue
import torch


os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


"""Parse command line arguments."""
argparser = argparse.ArgumentParser()
argparser.add_argument('--use_cuda', type=bool, default=False, help='Whether to use CUDA')
argparser.add_argument('--remove_duplicate_data', type=bool, default=False)
argparser.add_argument('--seq_len', type=int, default=15)
argparser.add_argument('--max_seq_len', type=int, default=6)
argparser.add_argument('--batch_size', type=int, default=4)
argparser.add_argument('--sample_size', type=int, default=1e7)
argparser.add_argument('--data', type=str, default='ddn_train')
argparser.add_argument('--save_dir', type=str, default='./sample')
argparser.add_argument('--save_file', type=str, default='sample_seq_1e-9.txt')
args = argparser.parse_args()


itos = pickle.load(open("./itos.bin", "rb"))
stoi = pickle.load(open("./stoi.bin", "rb"))


def select_top_char_by_threshold(char_prob, threshold):
    """
    :param char_prob: Tensor containing character probabilities.
    :param threshold: Probability threshold for character selection.
    :return: Set of characters that meet the threshold.
    """
    indices = torch.nonzero(char_prob >= threshold).view(-1).tolist()
    char_set = set(tuple([itos[x]]) for x in indices)
    return char_set


def get_next_char_prob(model, current_prefix, use_cuda):
    """
    :param model: The neural network model for character prediction.
    :param current_prefix: Current sequence of characters to condition on.
    :param use_cuda: Whether to use CUDA for computation.
    :return: Tensor containing the probability distribution over the next character.
    """
    z_inputs = torch.zeros(1, 1).long()
    hiddens = model.init_hiddens(1, use_cuda)
    if use_cuda:
        z_inputs = z_inputs.cuda()
        hiddens = hiddens.cuda()

    _, hiddens = model(z_inputs, hiddens)
    if current_prefix is not None:
        for char in current_prefix:
            z_inputs[0, 0] = stoi[char]
            output, hiddens = model(z_inputs, hiddens)
    else:
        output, hiddens = model(z_inputs, hiddens)
    return output.squeeze()



def beam_search(model, seq_len, alpha, save_batch, use_cuda):
    """
    :param model: The neural network model for sequence generation.
    :param seq_len: Maximum sequence length to generate.
    :param alpha: Probability threshold for pruning the search space.
    """
    lookup_table = {}
    vocab_size = len(itos)
    prefixes = queue.Queue()
    generate_datas = []
    have_saved_batch = 0
    max_len = -1
    start_time = time.time()

    char_set = []

    next_char_prob = get_next_char_prob(model, None, use_cuda)
    for i in range(2, vocab_size):
        lookup_table[tuple([itos[i]])] = 1.0 * next_char_prob[i].item()

    # 0 represents <begin> and 1 represents <end>
    for i in range(2, vocab_size):
        # use [itos[x]] rather than itos[x] to deal the case that x is a <begin> and <end> token
        prefixes.put(tuple([itos[i]]))
        char_set.append(tuple([itos[i]]))
    char_set.append(tuple([itos[0]]))
    char_set.append(tuple([itos[1]]))
    char_set = tuple(char_set)

    while not prefixes.empty():
        current_prefix = prefixes.get()
        next_char_prob = get_next_char_prob(model, current_prefix, use_cuda)
        for char in char_set:
            next_prob = lookup_table[current_prefix] * next_char_prob[stoi[char[0]]].item()
            if next_prob > alpha:
                if char[0] == '<end>':
                    if len(current_prefix) >= 6:
                        generate_datas.append(current_prefix)
                    else:
                        continue
                elif len(current_prefix) >= seq_len:
                    generate_datas.append(current_prefix)
                else:
                    if max_len < len(current_prefix):
                        max_len = len(current_prefix)
                        end_time = time.time()
                        run_time = end_time - start_time
                        print("max_len: {}, run_time: {}".format(max_len, run_time))
                    prefixes.put(current_prefix + char)
                    lookup_table[current_prefix + char] = next_prob
        if len(generate_datas) >= save_batch:
            have_saved_batch += 1
            print("have_saved_batch: {}".format(have_saved_batch))
            with open(os.path.join(args.save_dir, args.save_file), "a") as f:
                for data in generate_datas:
                    save_str = ""
                    for char in data:
                        save_str += char
                    f.write(f"{save_str} {lookup_table[data]}" + "\n")
                f.flush()
                f.close()
            generate_datas = []
    with open(os.path.join(args.save_dir, args.save_file), "a") as f:
        for data in generate_datas:
            save_str = ""
            for char in data:
                save_str += char
            f.write(f"{save_str} {lookup_table[data]}" + "\n")
        f.flush()
        f.close()



def main():
    batch_size = args.batch_size
    seq_len = args.seq_len
    sample_size = args.sample_size
    use_cuda = args.use_cuda


    model = torch.load("save/to/Guesser.pth")
    model.eval()

    if args.use_cuda:
        model = model.cuda()

    beam_search(model=model,
                seq_len=seq_len,
                alpha=1e-9,
                save_batch=1e5,
                use_cuda=use_cuda)


if __name__ == "__main__":
    main()
