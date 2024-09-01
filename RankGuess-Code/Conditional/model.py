import torch
import torch.nn.init as init

class Guesser(torch.nn.Module):
    
    def __init__(self, vocab_size) -> None:
        super(Guesser, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_layer = torch.nn.Embedding(vocab_size, 128)
        self.rnn_layer = torch.nn.GRU(128, 256, batch_first=True)
        self.linear_layer = torch.nn.Linear(256, vocab_size)
        self.dropout = torch.nn.Dropout(0.6)

        self.initialize_weights()

    def initialize_weights(self):
        init.xavier_uniform_(self.embedding_layer.weight)
        init.xavier_uniform_(self.rnn_layer.weight_ih_l0)
        init.xavier_uniform_(self.rnn_layer.weight_hh_l0)
        init.constant_(self.rnn_layer.bias_ih_l0, 0)
        init.constant_(self.rnn_layer.bias_hh_l0, 0)
        init.xavier_uniform_(self.linear_layer.weight)

    def forward(self, inputs):
        embedded = self.embedding_layer(inputs)
        outputs, _ = self.rnn_layer(embedded)
        outputs = self.linear_layer(outputs)
        outputs = self.dropout(outputs)
        outputs = torch.nn.functional.gumbel_softmax(outputs, tau=10, dim=-1)
        return outputs


class Ranker(torch.nn.Module):

    def __init__(self, vocab_size) -> None:
        super(Ranker, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_layer = torch.nn.Embedding(vocab_size, 128)
        self.rnn_layer = torch.nn.GRU(128, 256, batch_first=True)
        self.linear_layer = torch.nn.Linear(256, 64)
        self.relu = torch.nn.LeakyReLU()
        self.dropout = torch.nn.Dropout(0.4)

        self.initialize_weights()

    def initialize_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'rnn' in name:
                    torch.nn.init.kaiming_normal_(param)
                elif 'linear' in name:
                    torch.nn.init.kaiming_uniform_(param)
            elif 'bias' in name:
                torch.nn.init.zeros_(param)

    def forward(self, inputs):
        embedded = self.embedding_layer(inputs)
        outputs, _ = self.rnn_layer(embedded)
        outputs = self.linear_layer(outputs)
        outputs = self.relu(outputs)
        outputs = self.dropout(outputs)
        outputs = outputs.view(outputs.size(0), -1)
        return outputs


if __name__ == "__main__":
    Ranker = Ranker(96)
    input = torch.randint(low=0, high=96, size=(4, 18))
    output = Ranker(input)
    print(output.shape)