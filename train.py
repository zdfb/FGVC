import timm
import time
import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
import utils.DataSet_Aug as DataSet
from utils.LabelSmooth import LabelSmoothCELoss


###### Function: Train the samples after augmentation ######

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', choices=['CUB_200_2011', 'NABirds', 'Cars',
                        'Dogs', 'Flowers'], default='CUB_200_2011', help='Which dataset')
    parser.add_argument('--datapath', type=str,
                        default='data/CUB_200_2011', help='The root path of dataset.')
    parser.add_argument('--savepath', type=str, default='./vit_cub.pth',
                        help='The save path of output model.')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr_c', type=float, default=0.001,
                        help='The initial learning rate of classifier.')
    parser.add_argument('--lr_f', type=float, default=0.0002,
                        help='The initial learning rate of features.')
    parser.add_argument('--momentum', type=float,
                        default=0.8, help='The momuntum of SGD.')
    parser.add_argument('--weight_decay', type=float,
                        default=5e-4, help='The weight decay.')
    parser.add_argument('--step_size', type=int, default=30,
                        help='We adopt the StepLR.')
    parser.add_argument('--gamma', type=float, default=0.3,
                        help='The gamma of StepLR.')
    parser.add_argument('--smoothing', type=float, default=0.4,
                        help='The parameter of label-smoothing.')
    args = parser.parse_args()

    return args


def train(args):

    # hyper parameters
    DATA_SET = args.dataset
    DATA_PATH = args.datapath
    SAVE_PATH = args.savepath
    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    LEARING_RATE_CLASSIFIER = args.lr_c
    LEARING_RATE_FEATURES = args.lr_f
    MOMENTUM = args.momentum
    WEIGHT_DECAY = args.weight_decay
    STEP_SIZE = args.step_size
    STEP_GAMMA = args.gamma
    SMOOTHING = args.smoothing

    # define the trainloader and testloader
    train_loader = torch.utils.data.DataLoader(
        DataSet.load_datasets(dataset=DATA_SET, root=DATA_PATH,
                              train=True, transform=DataSet.data_transform['train']),
        batch_size=BATCH_SIZE, shuffle=True,
        num_workers=2)  # load the trainset

    test_loader = torch.utils.data.DataLoader(
        DataSet.load_datasets(dataset=DATA_SET, root=DATA_PATH,
                              train=False, transform=DataSet.data_transform['test']),
        batch_size=BATCH_SIZE, shuffle=False,
        num_workers=2)  # load the testset

    # define the model
    net = timm.create_model('vit_base_patch16_384',
                            pretrained=True)
    inchannel = net.head.in_features

    if DATA_SET == 'CUB_200_2011':
        class_num = 200
    elif DATA_SET == 'NABirds':
        class_num = 555
    elif DATA_SET == 'Cars':
        class_num = 196
    elif DATA_SET == 'Dogs':
        class_num = 120
    elif DATA_SET == 'Flowers':
        class_num = 102

    net.head = nn.Linear(inchannel, class_num)
    net.to(device)

    # define the lossfunction and optimzer
    labelSmoothCELoss = LabelSmoothCELoss(SMOOTHING)

    # using differential learning rate strategy
    classifier_params = []
    features_params = []
    for name, params in net.named_parameters():
        if 'head' in name:
            classifier_params += [params]
        else:
            features_params += [params]

    optimizer = optim.SGD(
        params=[
            {"params": classifier_params, 'lr': LEARING_RATE_CLASSIFIER},
            {"params": features_params},
        ],
        lr=LEARING_RATE_FEATURES, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE,
                                          gamma=STEP_GAMMA, last_epoch=-1)

    # train and test
    print(EPOCHS)
    best_accuracy = 0.0

    for epoch in range(EPOCHS):

        epoch_start = time.time()
        print('Epoch:{}'.format(epoch + 1))

        # train
        net.train()
        train_loss_list = []  # record the loss of every batch
        train_accuracy_list = []  # record the accuracy of every batch

        for step, data in enumerate(train_loader, start=0):

            images, labels = data
            images, labels = images.to(device), labels.to(
                device)  # labels.long()
            logits = net(images)
            loss = labelSmoothCELoss(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss = loss.item()
            train_loss_list.append(train_loss)

            prediction = torch.max(logits, dim=1)[-1]
            train_accuracy = prediction.eq(labels).cpu().float().mean()
            train_accuracy_list.append(train_accuracy)

            rate = (step + 1) / len(train_loader)
            a = "*" * int(rate * 50)
            b = "." * int((1 - rate) * 50)
            print("\rtrain loss: {:^3.0f}%[{}->{}]{:.3f}".format(
                int(rate * 100), a, b, loss), end="")  # draw the progress bar
        print()
        print('train_loss:{:.3f},train_accuracy:{:.3f}'.format(
            np.mean(train_loss_list), np.mean(train_accuracy_list)))

        # test
        net.eval()
        test_loss_list = []
        test_accuracy_list = []

        with torch.no_grad():
            for step, data in enumerate(test_loader, start=0):

                images, labels = data
                images, labels = images.to(device), labels.to(device)
                logits = net(images)
                loss = labelSmoothCELoss(logits, labels)

                test_loss = loss.item()
                test_loss_list.append(test_loss)

                prediction = torch.max(logits, dim=1)[-1]
                test_accuracy = prediction.eq(labels).cpu().float().mean()
                test_accuracy_list.append(test_accuracy)

                rate = (step + 1) / len(test_loader)
                a = "*" * int(rate * 50)
                b = "." * int((1 - rate) * 50)
                print("\rtest loss: {:^3.0f}%[{}->{}]{:.3f}".format(
                    int(rate * 100), a, b, loss), end="")
            print()

            test_accuracy = np.mean(test_accuracy_list)
            if test_accuracy > best_accuracy:
                best_accuracy = test_accuracy
                torch.save(net.state_dict(), SAVE_PATH)

            epoch_end = time.time()
            print('test_loss:{:.3f},test_accuracy:{:.3f},epoch_time:{:.3f}'.format(
                np.mean(test_loss_list), np.mean(test_accuracy_list), (epoch_end-epoch_start)))
        scheduler.step()

    print('Finished Training')
    print('The best accuracy : %.3f' % best_accuracy)


def main():
    args = parse_args()
    train(args)


if __name__ == '__main__':
    main()
