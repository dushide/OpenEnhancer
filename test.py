import argparse
import copy
import os

import scipy
import torch.nn as nn
import warnings
import sys
import random
import numpy as np
import torch
from sklearn import metrics

from evaluation_metrics import EvaluationMetrics
from config import load_config
from loadMatData import load_data
import scipy  as sp
from label_utils import reassign_labels, special_train_test_split
from data import Multi_view_data, generate_partition
from models import DBONet

np.set_printoptions(threshold=np.inf)
import torch.nn.functional as F


def mixup_data(x, y, alpha=0.3, use_cuda=True):
    lam = np.random.beta(alpha, alpha)
    batch_size = y.size()[0]
    sample = 0
    trynum = 0
    while (sample < batch_size / 2 and trynum < 10):
        if use_cuda:
            index = torch.randperm(batch_size).cuda()
        else:
            index = torch.randperm(batch_size)
        #   剔除同类对
        index_dc = torch.nonzero(y[index] != y).squeeze()
        sample = len(index_dc)
        trynum += 1
    mixed_x = {}
    mixed_x[0] = lam * x[0][index_dc] + (1 - lam) * x[0][index][index_dc]
    for i in range(1, len(x)):
        if use_cuda:
            index2 = torch.randperm(batch_size).cuda()
        else:
            index2 = torch.randperm(batch_size)
        mixed_x[i] = lam * x[i][index_dc] + (1 - lam) * x[i][index2][index_dc]
    return mixed_x, y[index_dc], lam


def main(args, device):
    def valid(model, loader, device):
        model.eval()
        with torch.no_grad():
            label = []
            test_z = []
            correct, num_samples = 0, 0
            for batch in loader:
                x, y = batch['x'], batch['y']
                for k in x.keys():
                    x[k] = x[k].to(device)

                evidence = model(x)
                test_z.append(evidence.cpu().numpy())
                num_samples += len(batch['y'])
                label.append(batch['y'].cpu().numpy())
        test_z = np.concatenate(test_z)
        label = np.concatenate(label)
        test_truth_label_open = [NCLASSES if i == unseen_label_index else i for i in label]
        softmax_ccr, softmax_fpr, softmax_ccrs = EvaluationMetrics.ccr_at_fpr(np.array(test_truth_label_open),
                                                                              test_z,
                                                                              NCLASSES)

        return softmax_ccrs[-1]

    def train(model, train_loader, valid_loader, device):
        model = model.to(device)

        optimizer = torch.optim.Adam([
            {'params': (p for n, p in model.named_parameters() if p.requires_grad and 'weight' in n),
             'weight_decay': 1e-2},
            {'params': (p for n, p in model.named_parameters() if p.requires_grad and 'weight' not in n)},
        ], lr=args.learning_rate)
        step_lr = torch.optim.lr_scheduler.StepLR(optimizer, step_size=23, gamma=0.1)
        loss_list = list()
        best_valid_ccr = 0.0
        criterion = torch.nn.CrossEntropyLoss()
        for epoch in range(1, args.num_epoch + 1):
            model.train()
            train_loss, correct, num_samples = 0, 0, 0
            train_pred_label = []
            train_Z = []
            train_y = []
            output = []
            for batch in train_loader:
                x, y, index = batch['x'], batch['y'], batch['index']
                for k in x.keys():
                    x[k] = x[k].to(device)
                y = y.long().to(device)

                x_mix, y_mix, lam = mixup_data(x, y)
                x_all = {}
                for i in range(len(x)):
                    x_all[i] = torch.cat([x[i], x_mix[i]], dim=0)

                evidence = model(x_all)

                prob = evidence[:y.shape[0]]

                loss_Oen = 0
                for c in range(num_classes):
                    target_c = torch.LongTensor(y_mix.shape[0]).random_(c, c + 1).to(device)
                    loss_Oen += criterion(evidence[y.shape[0]:], target_c)

                loss_ce = criterion(evidence[:y.shape[0]], y)
                loss = args.lambda1 * loss_ce + args.lambda2 * (loss_Oen / num_classes)

                optimizer.zero_grad()

                loss.backward()
                optimizer.step()
                train_loss += loss.item()

                train_pred_label.append(y.cpu().detach().numpy())
                train_y.append(torch.cat([y], dim=0).cpu().detach().numpy())
                train_Z.append(evidence.cpu().detach().numpy())
                output.append(prob.cpu().detach().numpy())

                num_samples += len(y)
                correct += torch.sum(prob.argmax(dim=-1).eq(y)).item()
                print(f'Train-Epoch: {epoch} /{len(train_loader)}   Loss {loss.item()}\t')

            train_loss = train_loss
            train_acc = correct / num_samples
            loss_list.append(train_loss)
            valid_ccr = valid(model, valid_loader, device)
            if valid_ccr:
                if best_valid_ccr <= valid_ccr:
                    best_valid_ccr = valid_ccr
                    best_model_wts = copy.deepcopy(model.state_dict())
            else:
                valid_ccr = 0.0
            step_lr.step()
            print(
                f'Epoch {epoch:3d}: lr {step_lr.get_last_lr()[0]:.6f}, train loss: {train_loss:.4f}, train acc: {train_acc:.4f}, valid ccr: {valid_ccr:.4f}')
        try:
            model.load_state_dict(best_model_wts)
        except Exception as e:
            print(e)
        finally:
            test_Z, softmax_scores, test_truth_label = test(model, test_loader, device)
            test_truth_label_open = [NCLASSES if i == unseen_label_index else i for i in test_truth_label]
            softmax_pred_label = np.argmax(softmax_scores, 1)
            softmax_ACC = metrics.accuracy_score(test_truth_label_open,
                                                 softmax_pred_label)
            softmax_ccr, softmax_fpr, softmax_ccrs = EvaluationMetrics.ccr_at_fpr(np.array(test_truth_label_open),
                                                                                  test_Z,
                                                                                  NCLASSES)

            with open(args.file, "a") as f:
                f.write('batch_size:{},epoch:{},openness:{}, unseen_num:{},block:{},lambda1:{},lambda2:{}\n'.format(
                    args.batch_size, epoch, args.openness, unseen_num, args.layers, args.lambda1,
                    args.lambda2))
                f.write(' dataset:{},softmax_acc:{},softmax_ccrs:{}\n'.format(
                    data, round(
                        softmax_ACC * 100,
                        2), softmax_ccrs))
                f.write('loss:{}\n'.format(loss_list))
            if args.saveResp:
                print("save %s/%s_epoch%d.mat......" % (args.path, data, epoch))
                data_path = "%s/" % (args.path)
                if not os.path.exists(data_path):
                    os.makedirs(data_path)
                scipy.io.savemat("%s/%s_epoch%d.mat" % (data_path, data, epoch),
                                 {'test_indices': test_indices,
                                  'test_y': test_truth_label_open, 'test_Z': test_Z,
                                  'label_open': test_truth_label_open, 'label': test_truth_label,
                                  'softmax_ccr': softmax_ccr.tolist(), 'softmax_fpr': softmax_fpr.tolist(),
                                  })

    def test(model, loader, device):
        model.eval()
        with torch.no_grad():
            test_Z = []
            test_prob = []
            label = []
            correct, num_samples = 0, 0
            for batch in loader:
                x, y = batch['x'], batch['y']
                for k in x.keys():
                    x[k] = x[k].to(device)

                evidence = model(x)
                prob = F.softmax(evidence, 1)
                test_Z.append(evidence.cpu().numpy())
                test_prob.append(prob.cpu().numpy())
                label.append(batch['y'].cpu().numpy())
                pred_y = prob.argmax(dim=-1)
                correct += torch.sum(pred_y.cpu().eq(batch['y'])).item()
                num_samples += len(batch['y'])
        label = np.concatenate(label)
        prob = np.concatenate(test_prob)
        test_Z = np.concatenate(test_Z)
        return test_Z, prob, label

    thre = 0.01

    model = DBONet(n_feats, n_view, num_classes, args.layers, thre, device, fusion_type)

    print('---------------------------- Experiment ------------------------------')
    print('Number of views:', len(train_data.x), ' views with dims:', [v.shape[-1] for v in train_data.x.values()])
    print('Number of training samples:', len(train_data))
    print('Number of validating samples:', len(valid_data))
    print('Trainable Parameters:')
    for n, p in model.named_parameters():
        print('%-40s' % n, '\t', p.data.shape)
    print('----------------------------------------------------------------------')
    train(model, train_loader, valid_loader, device)


def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                        help='input batch size for training [default: 100]')
    parser.add_argument('--num_epoch', type=int, default=100, metavar='N',
                        help='number of epochs to train [default: 500]')
    parser.add_argument('--seed', type=int, default=20)
    parser.add_argument('--learning_rate', type=float, default=0.01, metavar='LR',
                        help='learning rate')

    parser.add_argument('--lambda1', type=float, default=1)
    parser.add_argument('--lambda2', type=float, default=0.1)
    parser.add_argument('--openness', type=float, default=0.1)
    parser.add_argument("--save_file", type=str, default="res_TMC0312.txt")
    parser.add_argument('--save_results', default=True)
    parser.add_argument('--fix_seed', default=True)
    parser.add_argument('--T', default=10)
    args = parser.parse_args()

    args.device = '0'
    device = torch.device('cpu' if args.device == 'cpu' else 'cuda:' + args.device)

    # tab_printer(args)

    dataset_dict = {1: 'esp_game', 2: 'MITIndoor', 3: 'NUSWIDEOBJ', 4: 'UCI', 5: 'Youtube', 6: "YTF10"}

    select_dataset = [1, 2, 3, 4, 5, 6]

    training_rate = 0.1
    valid_rate = 0.1
    unseen_label_index = -100

    # for  training_rate in training_rates:
    args.saveResp =False

    fusion_type='average'
    # fusion_type='weight'
    fusion_type = 'attention'
    args.file = f'./result/res_{fusion_type}_openness{args.openness}_layers.txt'
    args.file = f'./result/res_{fusion_type}_openness{args.openness}_lambda.txt'
    args.file = f'./result/res_{fusion_type}_openness{args.openness}.txt'
    args.file = f'./result/res_{fusion_type}_openness{args.openness}_ablation.txt'

    args.path  = './result/openness%.1f' % (args.openness)
    lambda1s = lambda2s = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1]


    # for batchsize in batchsizes:
    #     args.batch_size=batchsize
    #     file = './result/block_openness%.1f_batch=%d.txt' % (openness, args.batch_size)

    for ii in select_dataset:
        data = dataset_dict[ii]
        args.data = data
        config = load_config('./config/' + data)
        args.layers = config['block']
        args.lambda1=config['lambda1']
        args.lambda2=config['lambda2']

        features, labels = load_data(dataset_dict[ii], 'E:/code/data/')
        n_view = len(features)
        n_feats = [x.shape[1] for x in features]
        n = features[0].shape[0]
        n_classes = len(np.unique(labels))

        print(data, n, n_view, n_feats)

        open2 = (1 - args.openness) * (1 - args.openness)
        unseen_num = round((1 - open2 / (2 - open2)) * n_classes)
        print("unseen_num:%d" % unseen_num)

        original_num_classes = len(np.unique(labels))
        seen_labels = list(range(original_num_classes - unseen_num))
        y_true = reassign_labels(labels, seen_labels, unseen_label_index)

        train_indices, test_valid_indices = special_train_test_split(y_true, unseen_label_index,
                                                                     test_size=1 - training_rate)
        valid_indices, test_indices = generate_partition(y_true[test_valid_indices], test_valid_indices,
                                                         (valid_rate) / (1 - training_rate))
        num_classes = np.max(y_true) + 1
        NCLASSES = num_classes
        print('data:{}\tseen_labels:{}\trandom_seed:{}\tunseen_num:{}\tnum_classes:{}'.format(
            data,
            seen_labels,
            args.seed,
            unseen_num,
            num_classes))

        train_data = Multi_view_data(n_view, train_indices, features, y_true)
        valid_data = Multi_view_data(n_view, valid_indices, features, y_true)
        test_data = Multi_view_data(n_view, test_indices, features, y_true)

        train_loader = torch.utils.data.DataLoader(train_data
                                                   , batch_size=args.batch_size, shuffle=True)
        valid_loader = torch.utils.data.DataLoader(valid_data
                                                   , batch_size=args.batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_data
                                                  , batch_size=args.batch_size, shuffle=False)

        labels = torch.from_numpy(labels).long().to(device)
        y_true = torch.from_numpy(y_true).to(device)
        train_indices = torch.LongTensor(train_indices).to(device)

        N_mini_batches = len(train_loader)
        print('The number of training images = %d' % N_mini_batches)
        args.num_classes = num_classes
        args.seen_labels = seen_labels
        args.unseen_label_index = unseen_label_index
        # for lambda1 in lambda1s:
        #     for lambda2 in lambda2s:
        #         args.lambda1=lambda1
        #         args.lambda2=lambda2
        # for i in range(1,11):
        #         args.layers = i
        if args.fix_seed:
            torch.cuda.manual_seed(args.seed)  # 为当前GPU设置随机种子
            torch.cuda.manual_seed_all(args.seed)  # 为所有GPU设置随机种子
            random.seed(args.seed)
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)

        print("========================", data)

        main(args, device)
        with open(args.file, "a") as f:
            f.write('\n')
