
import torch
from ares.model.pytorch_wrapper import pytorch_classifier_with_logits
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))



def get_model_tf(cfg):#不使用cuda，因为攻击防御用到了tensorflow,与cuda不兼容
    model = network_set[cfg.network]()
    return model

def get_model_pytorch(cfg):#使用pytorch的cuda
    model = network_set[cfg.network]()
    model._inner = model._inner.to(device)
    return model


#两个网络都采用装饰器，为后续攻击防御提供了更多参数支持
#两个卷积，两个池化，三次激活，然后全连接调整至目标tensor
@pytorch_classifier_with_logits(n_class=10, x_min=0.0, x_max=1.0,x_shape=(1,28, 28), x_dtype=tf.float32, y_dtype=tf.int32)
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)

#两个网络都采用装饰器，为后续攻击防御提供了更多参数支持
#pytorch教学文档中提供的网络
#详见pytorch官方教学文档https://pytorch.org/tutorials/beginner/basics/intro.html
@pytorch_classifier_with_logits(n_class=1, x_min=0.0, x_max=1.0,x_shape=(1,28, 28), x_dtype=tf.float32, y_dtype=tf.int32)
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 16),
            nn.ReLU(),
            nn.Linear(16,16),
            nn.ReLU(),
            nn.Linear(16,16),
            nn.ReLU(),
            nn.Linear(16, 10),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

network_set={'Net':Net,'NeuralNetwork':NeuralNetwork}








def train(network,train_data,optimizer,loss_fn,epoch,train_losses,train_counter,tf=False):
    network.train()
    for batch_idx, (data, target) in enumerate(train_data):
        if not tf :
            data , target = data.to(device),target.to(device)
        optimizer.zero_grad()
        output = network(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
          print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(data), len(train_data.dataset),
            100. * batch_idx / len(train_data), loss.item()))
          train_losses.append(loss.item())
          train_counter.append(
            (batch_idx*64) + ((epoch-1)*len(train_data.dataset)))


def test(network,test_data,loss_fn,test_accurency,tf=False):
    network.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
      for data, target in test_data:
          if not tf:
            data , target = data.to(device),target.to(device)
          output = network(data)
          test_loss += loss_fn(output, target)
          correct += (output.argmax(1) == target).type(torch.float).sum().item()
    test_loss /= len(test_data.dataset)
    test_accurency.append(100.0 *correct / len(test_data.dataset))
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
      test_loss, correct, len(test_data.dataset),
      100. * correct / len(test_data.dataset)))
    
def test_adv(network,test_data,loss_fn,cfg,tf=False):
    network.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
      for data, target in test_data:
          if not tf:
            data , target = data.to(device),target.to(device)
          output = network(data)
          test_loss += loss_fn(output, target)
          correct += (output.argmax(1) == target).type(torch.float).sum().item()
    test_loss = test_loss/(len(test_data)*cfg.batch_size_test)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
      test_loss, correct, (len(test_data)*cfg.batch_size_test),
      100. * correct / (len(test_data)*cfg.batch_size_test)))

test_def = test_adv
