from data_process import *
from utility import *
from model import *
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import transforms


if __name__ == '__main__':
    # data process: there we only augment the train and validating data, instead of test data
    train_x, train_y, test_x, test_y = get_data('./expression.csv')

    # hyper parameters:
    batch_size = 64
    num_workers = 4

    mu, std = calculate_mean_std(train_x)

    # important part: data augmentation
    basic_trans = transforms.Compose([
        transforms.ToTensor(),
    ])
    train_augment = transforms.Compose([
        transforms.RandomResizedCrop(48, scale=(0.8, 1.2)),
        transforms.RandomApply([
            transforms.RandomAffine(0, translate=(0.2, 0.2)),
            transforms.RandomPerspective(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=.5, contrast=.5, saturation=.5, hue=.3),
        ], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.TenCrop(40),
        transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
        transforms.Lambda(lambda crops: torch.stack([transforms.Normalize(mu, std)(crop) for crop in crops])),
        transforms.Lambda(lambda crops: torch.stack([transforms.RandomErasing(p=.4)(crop) for crop in crops])),
    ])
    test_trans = transforms.Compose([
        transforms.TenCrop(40),
        transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
        transforms.Lambda(lambda crops: torch.stack([transforms.Normalize(mu, std)(crop) for crop in crops])),
    ])
    train_set = custom_dataset(train_x, train_y, input_transform=basic_trans)
    test_set = custom_dataset(test_x, test_y, input_transform=basic_trans)
    train_loader = DataLoader(train_set, shuffle=True, batch_size=batch_size, num_workers=num_workers)
    test_loader = DataLoader(test_set, shuffle=False, batch_size=batch_size, num_workers=num_workers)

    # hyper parameters, easy for changing
    mom = .9
    max_epoch = 10
    wd = 1e-6
    lr = 1e-2
    min_lr = 1e-6
    device = 'cuda' if torch.cuda.is_available() else 'cpu'  # default cuda

    """ question 1: use deep learning method to construct a classifier for expression recognition """
    alter_net = alter_cls([4, 2, 1]).to(device)
    alpha = torch.FloatTensor([.6, 1., 1., .2, .8, .4, .6]).to(device)
    criterion = adaptive_loss(alpha, 2, .05, num_class)
    optimizer = optim.SGD(params=alter_net.parameters(), lr=lr, weight_decay=wd, momentum=mom, nesterov=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.75, patience=5, verbose=True)

    # train
    total_acc_list, total_loss_list = [], []
    train_acc_list, train_loss_list = train(alter_net, criterion, optimizer, train_loader, device, max_epoch, scheduler)
    total_acc_list.extend(train_acc_list)
    total_loss_list.extend(train_loss_list)
    draw_accuracy_loss_curve(total_acc_list, total_loss_list)

    # test
    train_pre = test2pre(alter_net, train_loader, device, is_tenCrop=False)
    train_acc = cal_acc(train_pre, train_y)
    print('train accuracy: {:.6f}'.format(train_pre))
    draw_confusion_matrix(train_pre, train_y)

    test_pre = test2pre(alter_net, test_loader, device, is_tenCrop=False)
    test_acc = cal_acc(test_pre, test_y)
    print('test accuracy: {:.6f}'.format(test_acc))
    draw_confusion_matrix(test_pre, test_y)
