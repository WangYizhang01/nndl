import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


device = "cuda:3" if torch.cuda.is_available() else "cpu"
# device = "cpu"
print(f"Using {device} device")


class RNN_scratch(nn.Module):
    'Simple Recurrent Network'
    def __init__(self, input_size, hidden_size, output_size) -> None:
        super(RNN_scratch, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.init_weights()

    def init_weights(self):
        self.U = Parameter(torch.randn(self.hidden_size, self.hidden_size))
        self.W = Parameter(torch.randn(self.hidden_size, self.input_size))
        self.b = Parameter(torch.randn(self.hidden_size, 1))
        self.V = Parameter(torch.randn(self.output_size, self.hidden_size))

    def forward(self, input):
        H = torch.tensor([]).to(device)
        Y = torch.randn(self.output_size, 1).to(device)
        H = torch.cat((H, input[:, 0].reshape(-1, 1)), dim=1) # H[0]取值可能有问题，若input_size、hidden_size不一致，H[0]设置成x[0]会有问题
        for i in range(input.shape[1]):
            cur_h = torch.sigmoid(torch.mm(self.U, H[:, i].reshape(-1, 1)) + torch.mm(self.W, input[:, i].reshape(-1, 1)) + self.b)
            H = torch.cat((H, cur_h.reshape(-1, 1)), dim=1)
            cur_y = torch.mm(self.V, cur_h)
            Y = torch.cat((Y, cur_y.reshape(-1, 1)), dim=1)
        return Y[:, 1:], H[:, 1:]


class LSTM_scratch(nn.Module):
    def __init__(self, input_size, hidden_size, output_size) -> None:
        super(LSTM_scratch, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.init_weights()

    def init_weights(self):
        for suf in ['c', 'o', 'i', 'f']:
            setattr(self, 'U' + suf, Parameter(torch.randn(self.hidden_size, self.hidden_size)))
            setattr(self, 'W' + suf, Parameter(torch.randn(self.hidden_size, self.input_size)))
            setattr(self, 'b' + suf, Parameter(torch.randn(self.hidden_size, 1)))
    
    def forward(self, input):
        H = input[:, 0].reshape(-1, 1).to(device)
        C = input[:, 0].reshape(-1, 1).to(device) # 目前c的维度设置为hidden_size
        names = self.state_dict()
        for i in range(input.shape[1]):
            c_hat = torch.tanh(torch.mm(names['Uc'], H[:, i].reshape(-1, 1)) + torch.mm(names['Wc'], input[:, i].reshape(-1, 1)) + names['bc'])
            gate_o = torch.sigmoid(torch.mm(names['Uo'], H[:, i].reshape(-1, 1)) + torch.mm(names['Wo'], input[:, i].reshape(-1, 1)) + names['bo'])
            gate_i = torch.sigmoid(torch.mm(names['Ui'], H[:, i].reshape(-1, 1)) + torch.mm(names['Wi'], input[:, i].reshape(-1, 1)) + names['bi'])
            gate_f = torch.sigmoid(torch.mm(names['Uf'], H[:, i].reshape(-1, 1)) + torch.mm(names['Wf'], input[:, i].reshape(-1, 1)) + names['bf'])

            cur_c = gate_f * C[:, i].reshape(-1, 1) + gate_i * c_hat
            cur_h = gate_o * torch.tanh(cur_c)
            C = torch.cat((C, cur_c.reshape(-1, 1)), dim=1)
            H = torch.cat((H, cur_h.reshape(-1, 1)), dim=1)
        return H[:, 1:],


class GRU_scratch(nn.Module):
    def __init__(self, input_size, hidden_size, output_size) -> None:
        super(GRU_scratch, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.init_weights()

    def init_weights(self):
        for suf in ['z', 'r', 'h']:
            setattr(self, 'U' + suf, Parameter(torch.randn(self.hidden_size, self.hidden_size)))
            setattr(self, 'W' + suf, Parameter(torch.randn(self.hidden_size, self.input_size)))
            setattr(self, 'b' + suf, Parameter(torch.randn(self.hidden_size, 1)))
    
    def forward(self, input):
        H = input[:, 0].reshape(-1, 1).to(device)
        names = self.state_dict()
        for i in range(input.shape[1]):
            gate_z = torch.sigmoid(torch.mm(names['Uz'], H[:, i].reshape(-1, 1)) + torch.mm(names['Wz'], input[:, i].reshape(-1, 1)) + names['bz'])
            gate_r = torch.sigmoid(torch.mm(names['Ur'], H[:, i].reshape(-1, 1)) + torch.mm(names['Wr'], input[:, i].reshape(-1, 1)) + names['br'])
            h_hat = torch.tanh(torch.mm(names['Uh'], gate_r * H[:, i].reshape(-1, 1)) + torch.mm(names['Wh'], input[:, i].reshape(-1, 1)) + names['bh'])

            cur_h = gate_z * H[:, i].reshape(-1, 1) + (1 - gate_z) * h_hat
            H = torch.cat((H, cur_h.reshape(-1, 1)), dim=1)
        return H[:, 1:],


if __name__ == '__main__':
    input = torch.randn(10, 8).to(device)
    # model = RNN_scratch(10, 10, 5)
    # model = LSTM_scratch(10, 10, 5)
    model = GRU_scratch(10, 10, 5)
    model.to(device)
    print(model(input)[0])
