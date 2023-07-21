import argparse
import os
import cv2
import numpy as np
import pickle
import time
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from tqdm import tqdm

from model.ResNet_models import Model
from data import Train_Loader, Test_Loader
from utils import adjust_lr, AvgMeter, compute_entropy_individual, compute_bce_individual
from SOD_Evaluation_Tool.evaluator import Eval_thread
from SOD_Evaluation_Tool.dataloader import EvalDataset
from perturbance import *
from ECE import Compute_ECE_Valid, ECE_EW




def train_ASLP_MC(param):
    generator = Model(channel=param.feat_channel)
    generator.cuda()
    generator_params = generator.parameters()
    generator_optimiser = torch.optim.Adam(generator_params, param.lr_gen)

    train_loader = Train_Loader(param.train_image_root,
                                   param.train_gt_root,
                                   batchsize = param.batchsize,
                                   trainsize = param.trainsize,
                                   name_file = param.train_name_file)

    Alpha = {}
    top_acc = torch.tensor(0.0)
    name_file = param.train_name_file
    with open(name_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            Alpha[line] = torch.tensor(0.0)

    for epoch in range(1, (param.epoch + 1)):
        generator.train()
        loss_record = AvgMeter()
        BCE_record = AvgMeter()
        CAL_record = AvgMeter()
        Alpha_record = AvgMeter()

        with tqdm(train_loader, unit='batch') as pbar:
            for pack in pbar:
                pbar.set_description('Epoch {}/{}'.format(epoch, param.epoch))
                generator_optimiser.zero_grad()
                images, gts, names = pack
                images = Variable(images)
                gts = Variable(gts)
                images = images.cuda()
                gts = gts.cuda()

                n = images.shape[0]
                al = torch.zeros(n).cuda()
                for i in range(n):
                    al[i] = Alpha[names[i]]

                pts = stochastic_label_perturbation(gt=gts, alpha=al, beta=param.beta, noise=param.perturb_noise)
                pred = generator(images)

                loss_all = F.binary_cross_entropy_with_logits(pred, pts, reduce='none')
                BCE_all = compute_bce_individual(pred.detach(), gts.detach())
                CAL_all = compute_entropy_individual(pred.detach())

                loss_all.backward()
                generator_optimiser.step()

                loss_record.update(loss_all.data, n)
                BCE_record.update(torch.mean(BCE_all).data, n)
                CAL_record.update(torch.mean(CAL_all).data, n)
                Alpha_record.update(torch.mean(al).data, n)
                pbar.set_postfix(Loss='{:.4f}'.format(loss_record.show().item()),
                                 BCE='{:.4f}'.format(BCE_record.show().item()),
                                 CAL='{:.4f}'.format(CAL_record.show().item()),
                                 alpha='{:.4f}'.format(Alpha_record.show().item()),
                                 lr=generator_optimiser.param_groups[0]['lr'])

                if epoch > 5:
                    zrs = torch.zeros_like(al)
                    # al = al + param.eta * ((- 2 * BCE_all + CAL_all) / param.beta +
                    #                         param.lbd * torch.minimum(((1.0 - cur_acc) / param.beta) - al, zrs))
                    al = al + param.eta * ((- 2 * BCE_all + CAL_all) / param.beta +
                                            param.lbd * torch.minimum((1.0 - cur_acc) - param.beta * al, zrs))

                    for i in range(n):
                        if al[i] < 0:
                            Alpha[names[i]] = torch.tensor(0.0)
                        elif al[i] >= 1 / (2 * param.beta):
                            Alpha[names[i]] = torch.tensor(1 / (2 * param.beta) - 0.01)
                        else:
                            Alpha[names[i]] = al[i].cpu()

        cur_acc = torch.tensor(acc_val(generator, param))
        print("Top Accuracy: {}, Current Accuracy: {}.".format(top_acc.item(), cur_acc.item()))

        if cur_acc > top_acc:
            top_acc = cur_acc

        adjust_lr(generator_optimiser, epoch, param.decay_rate, param.decay_epoch)

        save_path = './experiments'
        exp_group_path = os.path.join(save_path, param.exp_group)
        exp_path = os.path.join(exp_group_path, param.exp_name)

        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if not os.path.exists(exp_group_path):
            os.makedirs(exp_group_path)
        if not os.path.exists(exp_path):
            os.makedirs(exp_path)
        if epoch % param.epoch == 0:
            torch.save(generator.state_dict(), exp_path + '/' + 'Model' + '_%d' % epoch + '_Gen.pth')
            Alpha_sorted = dict(sorted(Alpha.items(), key=lambda item: item[1]))
            with open(exp_path + '/' + 'Alpha.pkl', 'wb') as f:
                pickle.dump(Alpha_sorted, f)



def acc_val(model, param):
    model = model.eval()
    save_path = './experiments/{}/{}/Validation/'.format(param.exp_group, param.exp_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    valid_loader = Test_Loader(image_root = param.train_image_root, 
                               testsize = param.testsize, 
                               name_file = param.valid_name_file)

    for i in tqdm(range(valid_loader.size), desc='Validation'):
        image, HH, WW, name = valid_loader.load_data()
        image = image.cuda()
        generator_pred = model.forward(image)
        res = generator_pred
        res = F.interpolate(res, size=[WW, HH], mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = 255 * res
        cv2.imwrite(save_path + name, res)

    dir_pred = './experiments/{}/{}/{}'.format(param.exp_group, param.exp_name, 'Validation')
    acc = Compute_ECE_Valid(dir_pred = dir_pred, 
                            dir_gt = param.train_gt_root, 
                            param = pm, 
                            n_bins = 10, 
                            ACC = True)

    return acc



def test(param):
    generator = Model(channel=param.feat_channel)
    generator.load_state_dict(torch.load('./experiments/{}/{}/Model_{}_Gen.pth'.format(param.exp_group, param.exp_name, param.epoch)))
    generator.cuda()
    generator.eval()
    test_datasets = ['DUTS-TE', 'DUT-OMRON', 'SOD', 'PASCAL-S', 'ECSSD', 'HKU-IS']
    for dataset in test_datasets:
        print('Currently processing: {} dataset:'.format(dataset))
        save_path = './experiments/{}/{}/{}/'.format(param.exp_group, param.exp_name, param.exp_name) + dataset + '/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        image_root = param.test_dataset_root + dataset + '/Image/'
        test_loader = Test_Loader(image_root, param.testsize)
        for i in tqdm(range(test_loader.size), desc='{}'.format(dataset)):
            image, HH, WW, name = test_loader.load_data()
            image = image.cuda()
            generator_pred = generator.forward(image)
            res = generator_pred
            res = F.interpolate(res, size=[WW, HH], mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = 255 * (res - res.min()) / (res.max() - res.min() + 1e-8)
            cv2.imwrite(save_path + name, res)
        print('Testing in {} dataset has been completely!'.format(dataset))


def eval(param):
    output_dir = './SOD_Evaluation_Tool/Result/Detail'
    pred_dir = './experiments/{}/{}/{}/'.format(param.exp_group, param.exp_name, param.exp_name)
    threads = []
    for dataset in pm.test_dataset:
        loader = EvalDataset(os.path.join(pred_dir, dataset), os.path.join(pm.test_dataset_root, dataset, 'GT'))
        thread = Eval_thread(loader, param.exp_name, dataset, output_dir, True)
        threads.append(thread)
    for thread in threads:
        print(thread.run())



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--epoch', type=int, default=30, help='epoch number')
    parser.add_argument('--lr_gen', type=float, default=2.5e-5, help='learning rate for generator')
    parser.add_argument('--batchsize', type=int, default=10, help='training batch size')
    parser.add_argument('--trainsize', type=int, default=384, help='training dataset size')
    parser.add_argument('--testsize', type=int, default=384, help='testing dataset size')
    parser.add_argument('--decay_rate', type=float, default=0.95, help='decay rate of learning rate')
    parser.add_argument('--decay_epoch', type=int, default=10, help='every n epochs decay learning rate')
    parser.add_argument('--feat_channel', type=int, default=256, help='reduced channel of saliency feat')
    parser.add_argument('--exp_group', type=str, default='Official', help='experiment group name')
    parser.add_argument('--exp_name', type=str, default='ASLP_MC_3', help='experiment name')
    parser.add_argument('--alpha', type=float, default=0, help='perturb rate of the groundtruth')
    parser.add_argument('--beta', type=float, default=2.0, help='perturbation hyperparameter')
    parser.add_argument('--perturb_noise', type=bool, default=False)
    parser.add_argument('--eta', type=float, default=0.002, help='gradient hyperparameter')
    parser.add_argument('--lbd', type=float, default=2000.0, help='strength of accuracy enforcer')
    parser.add_argument('--ada_bins', type=int, default=40)
    parser.add_argument('--start_epoch', type=int, default=5, help='start point of updating alpha')
    parser.add_argument('--train_image_root', default='./Dataset/Train/DUTS-TR/Image/')
    parser.add_argument('--train_gt_root', default='./Dataset/Train/DUTS-TR/GT/')
    parser.add_argument('--train_name_file', default='./Dataset/DUTS-TR-Train.txt')
    parser.add_argument('--valid_name_file', default='./Dataset//DUTS-TR-Validation.txt')
    parser.add_argument('--test_dataset_root', default='./Dataset/Test/')
    parser.add_argument('--test_dataset', type=list, default=['DUTS-TE', 'DUT-OMRON', 'PASCAL-S', 'SOD', 'ECSSD', 'HKU-IS'])
    pm = parser.parse_args()
    print(pm)

    # start = time.time()
    # train_ASLP_MC(param=pm)
    # end = time.time()
    # duration = (start - end) / 3600
    # print("Training time: {} hours".format(duration))

    # test(pm)

    # for dataset in pm.test_dataset:
    #     dir_pred = './experiments/{}/{}/{}/{}'.format(pm.exp_group, pm.exp_name, pm.exp_name, dataset)
    #     dir_gt = './Dataset/Test/{}/GT'.format(dataset)
    #     ECE_EW(dir_pred=dir_pred, dir_gt=dir_gt, param=pm, dataset=dataset, method=pm.exp_name, n_bins=10)

    eval(pm)


