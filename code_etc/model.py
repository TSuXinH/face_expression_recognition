import torch
import numpy as np
import torch.nn as nn
from PIL import Image
import torch.nn.functional as F


class basic_cls(nn.Module):
    """
    Create classification net using VGG backbone
    """
    def __init__(self):
        super(basic_cls, self).__init__()
        self.cbrp_1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        )
        self.cbrp_2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 128, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
        )
        self.cbrp_3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=(3, 3)),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
        )
        self.cbrp_4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
        )
        self.linear = nn.Sequential(
            nn.Linear(512 * 2 * 2, 256),
            nn.Dropout(.3),
            nn.ReLU(True),
            nn.Linear(256, 7),
        )

    def forward(self, input_data):
        temp = self.cbrp_1(input_data)
        temp = self.cbrp_2(temp)
        temp = self.cbrp_3(temp)
        temp = self.cbrp_4(temp)
        temp = temp.view(-1, 512 * 2 * 2)
        output = self.linear(temp)
        return output


class alter_cls(nn.Module):
    """
    Create classification net using resnet backbone
    """
    def __init__(self, block_chn):
        super(alter_cls, self).__init__()
        self.block_chn = block_chn
        self.base = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(5, 5)),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(.1),
            nn.Conv2d(32, 64, kernel_size=(5, 5)),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(.1),
        )
        self.residual_block_1 = create_res_layer([64, 128], self.block_chn[0])
        self.residual_block_2 = create_res_layer([128, 512], self.block_chn[1])
        self.residual_block_3 = create_res_layer([512, 128], self.block_chn[2])
        self.residual_block_4 = create_res_layer([128, 64], self.block_chn[3])
        self.linear_1 = nn.Sequential(
            nn.Linear(64 * 2 * 2, 512),
            nn.LeakyReLU(.1),
            nn.Dropout(.3),
        )
        self.linear_2 = nn.Sequential(
            nn.Linear(512, 128),
            nn.LeakyReLU(.1),
            nn.Dropout(.2),
        )
        self.linear_3 = nn.Linear(128, 7)
        self.init_weight()

    def init_weight(self):
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, 0, 1)
                nn.init.normal_(layer.bias, 0)
            elif isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='leaky_relu')
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

    def forward(self, input_batch):
        base_batch = self.base(input_batch)
        residual_batch = self.residual_block_1(base_batch)
        residual_batch = self.residual_block_2(residual_batch)
        residual_batch = self.residual_block_3(residual_batch)
        residual_batch = self.residual_block_4(residual_batch)
        linear_batch = residual_batch.view(-1, 64 * 2 * 2)
        linear_batch = self.linear_1(linear_batch)
        linear_batch = self.linear_2(linear_batch)
        output_pre = self.linear_3(linear_batch)
        return output_pre


class res_block(nn.Module):
    def __init__(self, channels):
        super(res_block, self).__init__()
        self.layer_1 = nn.Sequential(
            nn.Conv2d(channels[0], channels[1], kernel_size=(1, 1)),
            nn.BatchNorm2d(channels[1]),
            nn.LeakyReLU(.1),
        )
        self.layer_2 = nn.Sequential(
            nn.Conv2d(channels[1], channels[0], kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(channels[0]),
        )

    def forward(self, input_data):
        return nn.LeakyReLU(.1)(self.layer_2(self.layer_1(input_data)) + nn.Sequential()(input_data))


def create_res_layer(chn, input_block_num):
    layer = [
        nn.Conv2d(chn[0], chn[1], kernel_size=(3, 3),
                  stride=(2, 2), padding=1, bias=False),
        nn.BatchNorm2d(chn[1]),
        nn.LeakyReLU(.1),
    ]
    r_channels = chn[:: -1]
    for num in range(input_block_num):
        layer.append(res_block(r_channels))
    return nn.Sequential(*layer)


class adaptive_loss(nn.Module):
    """
    Perform label smoothing + focal loss + loss gradient weight.
    Note that parameter 'alpha' has type: torch.FloatTensor,
    'alpha' should be sent to device before training if the device is cuda.
    """
    def __init__(self, alpha, gamma, smoothing_rate, num_class):
        super(adaptive_loss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smoothing_rate = smoothing_rate
        self.num_class = num_class

    def forward(self, pre, gt):
        """
        pre shape: batch_size * class number
        gt shape: batch_size,
        """
        log_softmax = F.log_softmax(pre, dim=1)
        expanded_gt = F.one_hot(gt, self.num_class)
        smoothing_gt = torch.clamp(
            expanded_gt.type(torch.float),
            self.smoothing_rate / self.num_class,
            1 - self.smoothing_rate * (1 - 1 / self.num_class)
        )
        weight = self.alpha.gather(dim=0, index=gt)  # weight[i] = alpha[gt[i]]
        smoothing_ce_loss = - torch.sum(log_softmax * smoothing_gt, dim=1)
        pt = torch.exp(- smoothing_ce_loss)
        adap_loss = weight * (1 - pt) ** self.gamma * smoothing_ce_loss
        return adap_loss.mean()


def train(net, criterion, optimizer, train_loader, device, max_epoch, is_tenCrop=False, scheduler=None):
    acc_list = []
    loss_list = []
    for epoch in range(max_epoch):
        temp_acc, temp_loss = train_once(net, criterion, optimizer, train_loader, device, is_tenCrop)
        acc_list.append(temp_acc)
        loss_list.append(temp_loss)
        print('epoch: {}'.format(epoch + 1))
        print('train loss: {:.6f}, train accuracy: {:.6f}'.format(temp_loss, temp_acc))
        if scheduler:
            scheduler.step()
    return acc_list, loss_list


def train_once(net, criterion, optimizer, loader, device, is_tenCrop=False):
    net.train()
    train_acc = 0
    train_loss = .0
    sample_num = .0
    for data, label in loader:
        optimizer.zero_grad()
        data, label = data.to(device), label.to(device)
        if is_tenCrop:
            batch_size, n_crops, chn, height, weight = data.size()
            data = data.view(-1, chn, height, weight)
            label = torch.repeat_interleave(label, repeats=n_crops, dim=0)
        prediction = net(data)
        loss = criterion(prediction, label)
        loss.backward()
        optimizer.step()
        train_acc += torch.sum((torch.argmax(prediction, dim=1) == label).type(torch.int)).item()
        train_loss += loss.item()
        sample_num += len(label)
        if device == 'cuda':
            torch.cuda.empty_cache()
    return train_acc / sample_num, train_loss / sample_num


def test2pre(net, loader, device, is_tenCrop=False):
    """
    Return the result of predicted class numbers, shape: (test_dataset_length, ), type: nd-array
    the result shows class numbers, not the probabilities of each class
    """
    net.eval()
    output_pre = np.array([])
    with torch.no_grad():
        for data, _ in loader:
            data = data.to(device)
            if is_tenCrop:
                batch_size, n_crops, chn, height, weight = data.size()
                raw_pre = net(data.reshape(-1, chn, height, weight))
                temp = raw_pre.reshape(batch_size, n_crops, -1).mean(1)
            else:
                temp = net(data)
            pre = torch.argmax(temp, dim=1).cpu().numpy() if device == 'cuda' else torch.argmax(temp, dim=1).numpy()
            output_pre = np.concatenate([output_pre, pre], axis=0) if len(output_pre) else pre
    return output_pre


def quick_test(net, test_data, device, transforms, is_tenCrop=False):
    """
    Designed for the API in UI, this method gets rid of DataLoader.
    """
    net.eval()
    output_pre = np.array([])
    with torch.no_grad():
        for data in test_data:
            data = Image.fromarray(np.uint8(data))
            data = transforms(data)
            data = torch.unsqueeze(data, dim=0).to(device)
            if is_tenCrop:
                batch_size, n_crops, chn, height, weight = data.size()
                raw_pre = net(data.reshape(-1, chn, height, weight))
                temp = raw_pre.reshape(batch_size, n_crops, -1).mean(1)
            else:
                temp = net(data)
            pre = torch.argmax(temp, dim=1).cpu().numpy() if device == 'cuda' else torch.argmax(temp, dim=1).numpy()
            output_pre = np.concatenate([output_pre, pre], axis=0) if len(output_pre) else pre
    return output_pre


def test2prob(net, loader, device, is_tenCrop=False):
    """
    Return the result of predicted probability, shape: (test_dataset_length, num_class), type: nd-array
    Used for joint test.
    """
    net.eval()
    output_pre = np.array([])
    with torch.no_grad():
        for data, _ in loader:
            data = data.to(device)
            if is_tenCrop:
                batch_size, n_crops, chn, height, weight = data.size()
                raw_pre = net(data.reshape(-1, chn, height, weight))
                temp = raw_pre.reshape(batch_size, n_crops, -1).mean(1)
            else:
                temp = net(data)
            pre = temp.cpu().numpy() if device == 'cuda' else temp.numpy()
            output_pre = np.concatenate([output_pre, pre], axis=0) if len(output_pre) else pre
    return output_pre


def joint_test(model_list, ratio_list, loader, device, is_tenCrop=False):
    """
    Use more than one model to preform joint test.
    'ratio_list' is the weight showing how much each model will domain in the test.
    """
    for item in model_list:
        item.eval()
    with torch.no_grad():
        temp_list = []
        for item in model_list:
            temp_list.append(test2prob(item, loader, device, is_tenCrop))
        final_prob = np.zeros_like(temp_list[0])
        for idx, item in enumerate(temp_list):
            final_prob += ratio_list[idx] * item
        output_pre = np.argmax(final_prob, axis=1)
    return output_pre
