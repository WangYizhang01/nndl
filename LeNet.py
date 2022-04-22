from operator import le
from turtle import forward
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.nn.parameter import Parameter


# device = "cuda:3" if torch.cuda.is_available() else "cpu"
device = "cpu"
print(f"Using {device} device")

trainDataset = torchvision.datasets.MNIST(root='./data', train=True, transform=torchvision.transforms.ToTensor(), download=True)
testDataset = torchvision.datasets.MNIST(root='./data', train=False, transform=torchvision.transforms.ToTensor(), download=True)
trainDataloader = DataLoader(trainDataset, batch_size=64, shuffle=True)
testDataloader = DataLoader(testDataset, batch_size=1, shuffle=False)


class Conv2D:
    '''
    带连接表的二维卷积，支持单个样本
    '''
    def __init__(self, in_channels, out_channels, connect_nums, kernel_size=(3, 3), padding=0, stride=1) -> None:
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.connect_nums = connect_nums
        self.weights = []
        self.bias = []

    def conv2d(self, kernel, input):
        kernel_size = kernel.shape
        input = F.pad(input, [self.padding] * 4, 'constant')
        result = torch.zeros((input.shape[0] - kernel_size[0] + 2 * self.padding) // self.stride + 1,
                             (input.shape[1] - kernel_size[1] + 2 * self.padding) // self.stride + 1)

        for i in range(0, input.shape[0]-kernel_size[0]+1, self.stride):
            for j in range(0, input.shape[1]-kernel_size[1]+1, self.stride):
                result[i//self.stride, j//self.stride] = torch.sum(kernel * input[i: i + kernel_size[0], j: j + kernel_size[1]])
        
        return result

    def compute(self, X):
        assert X.shape[0] == self.in_channels
        output = torch.tensor([])
        for i in range(self.out_channels):
            start_index = 0
            cur_output = torch.tensor([])
            for j in range(start_index, start_index + self.connect_nums[i]):
                kernel = torch.randn(self.kernel_size, requires_grad=True)
                self.weights.append(kernel)
                cur_output = torch.cat((cur_output, self.conv2d(kernel, X[j % self.in_channels]).unsqueeze(0)), dim=0)
            wx = torch.sum(cur_output, dim=0)
            bias = torch.zeros_like(wx, requires_grad=True)
            self.bias.append(bias)
            output = torch.cat((output, (wx + bias).unsqueeze(0)), dim=0)
        return output


class BatchConv2D:
    '''
    带连接表的二维卷积，支持批次样本
    对每个输出channel的计算、卷积计算、对批数据的计算中均是采用循环方式，未实现并行，速度很慢
    '''
    def __init__(self, sample_shape, in_channels, out_channels, connect_nums_list, kernel_size=(3, 3), padding=0, stride=1) -> None:
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.connect_nums_list = connect_nums_list
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.weights = {}
        self.bias = []
        self.init_weights(sample_shape)

    def init_weights(self, sample_shape):
        for i in range(self.out_channels):
            start_index = 0
            for j in range(start_index, start_index + self.connect_nums_list[i]):
                kernel = torch.randn(self.kernel_size, requires_grad=True)
                if i in self.weights.keys():
                    self.weights[i].append(kernel)
                else:
                    self.weights[i] = [kernel]
            bias_shape = [(sample_shape[0] + 2 * self.padding - self.kernel_size[0]) // self.stride + 1,
                          (sample_shape[1] + 2 * self.padding - self.kernel_size[1]) // self.stride + 1]
            bias = torch.zeros(bias_shape, requires_grad=True)
            self.bias.append(bias)

    def conv2d(self, kernel, input):
        kernel_size = kernel.shape
        input = F.pad(input, [self.padding] * 4, 'constant')
        result = torch.zeros((input.shape[0] - kernel_size[0] + 2 * self.padding) // self.stride + 1,
                             (input.shape[1] - kernel_size[1] + 2 * self.padding) // self.stride + 1)

        for i in range(0, input.shape[0]-kernel_size[0]+1, self.stride):
            for j in range(0, input.shape[1]-kernel_size[1]+1, self.stride):
                result[i//self.stride, j//self.stride] = torch.sum(kernel * input[i: i + kernel_size[0], j: j + kernel_size[1]])
        
        return result

    def compute(self, X):
        assert X.shape[0] == self.in_channels
        assert len(self.connect_nums_list) == self.out_channels
        output = torch.tensor([])
        for i in range(self.out_channels):
            start_index = 0
            cur_output = torch.tensor([])
            for j in range(start_index, start_index + self.connect_nums_list[i]):
                # 第i个输出channel由第[start_index, start_index + connect_nums)个channel融合得到
                kernel = self.weights[i][j]
                cur_output = torch.cat((cur_output, self.conv2d(kernel, X[j % self.in_channels]).unsqueeze(0)), dim=0)
            wx = torch.sum(cur_output, dim=0)
            bias = self.bias[i]
            output = torch.cat((output, (wx + bias).unsqueeze(0)), dim=0)
        return output
    
    def batch_compute(self, batch_X):
        batch_output = torch.tensor([])
        for i in range(batch_X.shape[0]):
            output = self.compute(batch_X[i])
            batch_output = torch.cat((batch_output, output.unsqueeze(0)), dim=0)
        return batch_output


class BatchConv2D_(nn.Module):
    # TODO： 实现GPU运行
    '''
    带连接表的二维卷积，支持批次样本
    对每个输出channel的计算、卷积计算、对批数据的计算中均是采用循环方式，未实现并行，速度很慢
    '''
    def __init__(self, sample_shape, in_channels, out_channels, connect_nums_list, kernel_size=(3, 3), padding=0, stride=1) -> None:
        super(BatchConv2D_, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.connect_nums_list = connect_nums_list
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.weights = {}
        self.bias = []
        self.init_weights(sample_shape)
        # 将参数加入到Module.parameters()中
        # https://github.com/pytorch/pytorch/issues/76165
        # python动态变量
        for i in range(len(self.weights)):
            for j in range(len(self.weights[i])):
                setattr(self, 'weight_' + str(i) + str(j), Parameter(self.weights[i][j]))
        for i in range(len(self.bias)):
            setattr(self, 'bias_' + str(i), Parameter(self.bias[i]))

    def init_weights(self, sample_shape):
        for i in range(self.out_channels):
            start_index = 0
            for j in range(start_index, start_index + self.connect_nums_list[i]):
                kernel = torch.randn(self.kernel_size, requires_grad=True)
                if i in self.weights.keys():
                    self.weights[i].append(kernel)
                else:
                    self.weights[i] = [kernel]
            bias_shape = [(sample_shape[0] + 2 * self.padding - self.kernel_size[0]) // self.stride + 1,
                          (sample_shape[1] + 2 * self.padding - self.kernel_size[1]) // self.stride + 1]
            bias = torch.zeros(bias_shape, requires_grad=True)
            self.bias.append(bias)

    def conv2d(self, kernel, input):
        # TODO: 调用 F.conv2d
        kernel_size = kernel.shape
        input = F.pad(input, [self.padding] * 4, 'constant')
        result = torch.zeros((input.shape[0] - kernel_size[0] + 2 * self.padding) // self.stride + 1,
                             (input.shape[1] - kernel_size[1] + 2 * self.padding) // self.stride + 1)

        for i in range(0, input.shape[0]-kernel_size[0]+1, self.stride):
            for j in range(0, input.shape[1]-kernel_size[1]+1, self.stride):
                result[i//self.stride, j//self.stride] = torch.sum(kernel * input[i: i + kernel_size[0], j: j + kernel_size[1]])
        
        return result

    def compute(self, X):
        assert X.shape[0] == self.in_channels
        assert len(self.connect_nums_list) == self.out_channels
        output = torch.tensor([])
        names = self.__dict__
        for i in range(self.out_channels):
            start_index = 0
            cur_output = torch.tensor([])
            for j in range(start_index, start_index + self.connect_nums_list[i]):
                # 第i个输出channel由第[start_index, start_index + connect_nums)个channel融合得到
                kernel = names['weight_' + str(i) + str(j)]
                cur_output = torch.cat((cur_output, self.conv2d(kernel, X[j % self.in_channels]).unsqueeze(0)), dim=0)
            wx = torch.sum(cur_output, dim=0)
            bias = names['bias_' + str(i)]
            output = torch.cat((output, (wx + bias).unsqueeze(0)), dim=0)
        return output
    
    def forward(self, batch_X):
        # TODO: 实现批处理并行化，或许可采用高阶向量形式
        batch_output = torch.tensor([])
        for i in range(batch_X.shape[0]):
            output = self.compute(batch_X[i])
            batch_output = torch.cat((batch_output, output.unsqueeze(0)), dim=0)
        return batch_output


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.C1 = nn.Conv2d(1, 6, kernel_size=(5, 5))
        self.S2 = nn.AvgPool2d(kernel_size=2)
        self.C3 = nn.Conv2d(6, 16, kernel_size=(5, 5)) # 未使用连接表
        self.S4 = nn.AvgPool2d(kernel_size=2)
        self.C5 = nn.Conv2d(16, 120, kernel_size=(4, 4)) # mnist数据集为28*28，故需将kernel_size设置成4*4
        self.F6 = nn.Linear(120, 84)
        self.F7 = nn.Linear(84, 10)
        # self.model_list = nn.ModuleList([self.C1, self.S2, self.C3, self.S4, self.C5, self.F6, self.output])
    
    def forward(self, x):
        # for layer in self.model_list:
        #     x = layer(x)
        x = self.C1(x)
        x = self.S2(x)
        x = self.C3(x)
        x = self.S4(x)
        x = self.C5(x)
        x = x.reshape(-1, 120)
        x = self.F6(x)
        x = self.F7(x)
        y = nn.Softmax(dim=1)(x)
        # y = torch.argmax(y, dim=1)
        return y


class LeNet_withConnectedTable(nn.Module):
    def __init__(self):
        super(LeNet_withConnectedTable, self).__init__()
        self.C1 = nn.Conv2d(1, 6, kernel_size=(5, 5)) # 6*24*24
        self.S2 = nn.AvgPool2d(kernel_size=2) # 6*12*12
        self.C3 = BatchConv2D_((12, 12), 6, 16, [3]*16, kernel_size=(5, 5)) # 使用连接表        
        self.S4 = nn.AvgPool2d(kernel_size=2)
        self.C5 = nn.Conv2d(16, 120, kernel_size=(4, 4)) # mnist数据集为28*28，故需将kernel_size设置成4*4
        self.F6 = nn.Linear(120, 84)
        self.F7 = nn.Linear(84, 10)
   
    def forward(self, x):
        x = self.C1(x)
        x = self.S2(x)
        x = self.C3(x)
        x = self.S4(x)
        x = self.C5(x)
        x = x.reshape(-1, 120)
        x = self.F6(x)
        x = self.F7(x)
        y = nn.Softmax(dim=1)(x)
        return y


lr, epochs = 1e-4, 10
# lenet = LeNet()
lenet = LeNet_withConnectedTable()
lenet.to(device)
loss_fn = nn.CrossEntropyLoss()
optim = torch.optim.Adam(lenet.parameters(), lr=lr)


def train(epochs, net, dataloader, optim, loss_fn):
    net.train()
    for i in range(epochs):
        epoch_loss = 0.
        for batch, (X, y) in enumerate(dataloader):
            X = X.to(device); y = y.to(device)
            pred = net(X)
            loss = loss_fn(pred, y)
            epoch_loss += loss.item()

            optim.zero_grad()
            loss.backward()
            optim.step()
            if batch % 100 == 0:
                print(f'epoch {i} iter {batch} compeleted!')
        print('epoch %d: loss: %.6f' % (i, epoch_loss / len(dataloader)))

def test(net, dataloader, loss_fn):
    net.eval()
    epoch_loss = 0.
    count = 0
    for batch, (X, y) in enumerate(dataloader):
        X = X.to(device); y = y.to(device)
        pred = net(X)
        y_hat = torch.argmax(pred, dim=1)
        count += y_hat.item() == y.item()
        loss = loss_fn(pred, y)
        epoch_loss += loss.item()

    print('accracy %.6f: loss: %.6f' % (count / len(testDataset), epoch_loss / len(dataloader)))

if __name__ == '__main__':
    train(epochs, lenet, trainDataloader, optim, loss_fn)
    test(lenet, testDataloader, loss_fn)
