import os
import torch
import random
import numpy as np
import argparse
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from model import get_resnet
# from model1 import resnet50
from torchvision.models import resnet50
from torchvision import transforms
from dataset import CUB_dataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.utils.model_zoo import load_url


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-f37072fd.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-b627a593.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-0676ba61.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-63fe2227.pth'
}


def set_seed(seed):
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def select_device(device):
    cpu_request = device.lower() == 'cpu'

    if device and not cpu_request:
        os.environ['CUDA_VISIBLE_DEVICES'] = device
        assert torch.cuda.is_available(), 'CUDA is unavailable...'

    cuda = False if cpu_request else torch.cuda.is_available()
    return torch.device('cuda:0' if cuda else 'cpu')


def train(args):
    root_path = args.root_path
    root_image_path = os.path.join(root_path, 'images')
    assert os.path.exists(root_image_path), '{} root image path is not exists...'.format(root_image_path)
    assert os.path.exists(root_path), '{} root path is not exists...'.format(root_path)

    train_test_path = os.path.join(root_path, 'train_test_split.txt')
    images_txt_path = os.path.join(root_path, 'images.txt')
    images_labels_path = os.path.join(root_path, 'image_class_labels.txt')
    classes_txt_path = os.path.join(root_path, 'classes.txt')
    assert os.path.exists(train_test_path), '{} train_test_split.txt path is not exists...'.format(train_test_path)
    assert os.path.exists(images_txt_path), '{} image path is not exists...'.format(images_txt_path)
    assert os.path.exists(images_labels_path), '{} image_class_labels.txt path is not exists...'\
        .format(images_labels_path)
    assert os.path.exists(classes_txt_path), '{} classes.txt path is not exists...'.format(classes_txt_path)

    train_val_id = []
    test_id = []

    with open(train_test_path) as f:
        for line in f:
            image_id, is_train = line.split()
            if int(is_train) == 1:
                train_val_id.append(image_id)
            else:
                test_id.append(image_id)

    images_path = {}
    labels_dict = {}
    with open(images_txt_path) as f:
        for line in f:
            image_id, file_path = line.split()
            images_path[image_id] = file_path
    with open(images_labels_path) as f:
        for line in f:
            image_id, label = line.split()
            labels_dict[image_id] = label

    num_classes = 200
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    device = select_device(args.device)
    writer = SummaryWriter()

    train_image_path = []
    train_label = []
    test_image_path = []
    test_label = []
    for idx in train_val_id:
        train_image_path.append(images_path[idx])
        train_label.append(int(labels_dict[idx]) - 1)
    for idx in test_id:
        test_image_path.append(images_path[idx])
        test_label.append(int(labels_dict[idx]) - 1)

    print('train_val image num: {}'.format(len(train_image_path)))
    print('test image num: {}'.format(len(test_image_path)))

    data_transform = {
        'train': transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomResizedCrop((args.image_size, args.image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.43237721533416357, 0.49941621333172476, 0.4856074889829789],
                                 std=[0.2665100547329813, 0.22770540015765814, 0.2321024260764962])
        ]),
        'val': transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.43237713785804116, 0.49941626449353244, 0.48560741861744905],
                                 std=[0.2665100547329813, 0.22770540015765814, 0.2321024260764962])
        ])
    }

    nw = min([os.cpu_count(), args.batch_size if args.batch_size > 0 else 0, 8])

    train_dataset = CUB_dataset(root_image_path, train_image_path, train_label, data_transform['train'])
    test_dataset = CUB_dataset(root_image_path, test_image_path, test_label, data_transform['val'])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=nw)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=nw)

    depth = 50
    model_str = ''
    if depth == 18:
        model_str = 'resnet18'
    elif depth == 34:
        model_str = 'resnet34'
    elif depth == 50:
        model_str = 'resnet50'
    elif depth == 101:
        model_str = 'resnet101'

    model = get_resnet(depth, num_classes)
    # model = resnet50(num_classes)

    # model = resnet50(pretrained=False)
    # infeature = model.fc.in_features
    # model.fc = nn.Linear(infeature, num_classes)

    if args.pretrain:
        net_state_dict = model.state_dict()
        state_dict = load_url(model_urls[model_str], progress=True)
        pretrain_dict = {k: v for k, v in state_dict.items() if k in net_state_dict
                         and model.state_dict()[k].numel() == v.numel()}
        net_state_dict.update(pretrain_dict)
        model.load_state_dict(net_state_dict)

    if torch.cuda.device_count() > 1:
        print("Let's use {} GPUS".format(torch.cuda.device_count()))
        model = nn.DataParallel(model)

    model.to(device)

    lr = 0.01
    weight_decay = 1e-4
    if args.pretrain:
        milestones = [30, 60, 90]
    else:
        milestones = [250, 350, 400]

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

    criterion = nn.CrossEntropyLoss()

    train_loss_his = []
    train_acc_his = []
    test_loss_his = []
    test_acc_his = []

    best_acc = 0.0
    best_model_pth = './Model_Pth/resnet50_pretrain.pth'

    for epoch in range(args.epochs):

        train_epoch_loss = 0.0
        train_acc = 0.0
        model.train()
        train_bar = tqdm(train_loader)

        for step, (imgs, labels) in enumerate(train_bar):
            optimizer.zero_grad()
            imgs = imgs.to(device)
            labels = labels.to(device)

            out = model(imgs)
            loss = criterion(out, labels)

            train_epoch_loss += loss.item()
            predict = torch.max(out, dim=1)[1]

            train_acc += predict.eq(labels).sum()

            loss.backward()
            optimizer.step()

        scheduler.step()
        train_loss_his.append(train_epoch_loss / len(train_loader))
        train_acc_his.append(train_acc / len(train_dataset))
        writer.add_scalar('train/train loss', train_epoch_loss / len(train_loader), epoch+1)
        writer.add_scalar('train/train accuracy', train_acc / len(train_dataset), epoch+1)

        test_bar = tqdm(test_loader)
        test_epoch_loss = 0.0
        test_acc = 0.0
        model.eval()

        for step, (imgs, labels) in enumerate(test_bar):
            imgs = imgs.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                out = model(imgs)

            loss = criterion(out, labels)
            test_epoch_loss += loss.item()

            predict = torch.max(out, dim=1)[1]
            test_acc += predict.eq(labels).sum()

        test_acc_per = test_acc / len(test_dataset)
        test_loss_per = test_epoch_loss / len(test_loader)
        test_loss_his.append(test_loss_per)
        test_acc_his.append(test_acc_per)
        writer.add_scalar('test/test loss', test_loss_per, epoch+1)
        writer.add_scalar('test/test accuracy', test_acc_per, epoch+1)

        if test_acc_per > best_acc:
            best_acc = test_acc_per
            torch.save(model.state_dict(), best_model_pth)

        print('epoch[{}/{}]\ttrain loss: {:.4f}\ttrain acc: {:.4f}\ttest loss: {:.4f}\ttest acc: {:.4f}'
              .format(epoch + 1, args.epochs, train_epoch_loss / len(train_loader), train_acc / len(train_dataset),
                      test_epoch_loss / len(test_loader), test_acc / len(test_dataset)))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=32, help='the size of batch')
    parser.add_argument('--epochs', type=int, default=100, help='the num of epoch')
    parser.add_argument('--root-path', type=str, default='/datasets/CUB/CUB_200_2011',
                        help='the root path of CUB dataset')
    parser.add_argument('--image-size', type=int, default=224, help='the size of input image')
    parser.add_argument('--device', type=str, default='0', help='ie: 0 or 1 or 0,1 or cpu')
    parser.add_argument('--pretrain', action='store_true', help='using the pretrained weight')

    args = parser.parse_args()

    set_seed(0)
    train(args)


