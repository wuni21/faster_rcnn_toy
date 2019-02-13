import argparse
import torch
import os
import sys
import time
from tqdm import tqdm
from solver2 import Solver



parser = argparse.ArgumentParser(description='Pytorch implementation of Outlier detection using MCD and GAN')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--num_epoch', type=int, default=50, help='num_epoch')
parser.add_argument('--train_rpn', action='store_true', default=False, help='train rpn model')
parser.add_argument('--train_frcnn', action='store_true', default=False, help='train fast rcnn model')
parser.add_argument('--test', action='store_true', default=False, help='test model')
parser.add_argument('--checkpoint_dir', type=str, default='checkpoint', help='checkpoint dir')
args = parser.parse_args()

args.cuda = not args.no_cuda and torch.cuda.is_available()

print(args)

args.checkpoint_dir = 'checkpoint'
if not os.path.isdir(args.checkpoint_dir):
    os.mkdir(args.checkpoint_dir)


def train_rpn():
    solver = Solver(args)
    # solver.load_frcnn()
    start = time.time()
    time_list = [0]
    for i in tqdm(range(args.num_epoch), desc='Total'):
        print("********************"+"Epoch: "+str(i)+"***************************")

        '''train network'''
        print("===========================Training===========================")
        losses = solver.train(i, 'rpn')
        for _, key in enumerate(losses.keys()):
            print(str(key) + ': ' + str(torch.mean(torch.FloatTensor(losses[key]))))


        '''Remain time calculation'''
        time_list.append(time.time()-start-sum(time_list))
        time_per_epoch = sum(time_list[1:])/(len(time_list)-1)
        time_remain = (args.num_epoch-(i+1))*time_per_epoch
        hour = int(time_remain/3600)
        delta = time_remain-hour*3600
        min = int(delta/60)
        delta = delta-min*60
        sec = int(delta)
        print("{}h {}m {}s Left".format(hour, min, sec))
        solver.save_rpn()

def train_frcnn():
    solver = Solver(args)
    solver.load_rpn()
    start = time.time()
    time_list = [0]
    for i in tqdm(range(args.num_epoch), desc='Total'):
        print("********************"+"Epoch: "+str(i)+"***************************")

        '''train network'''
        print("===========================Training===========================")
        losses = solver.train(i, 'frcnn')
        for _, key in enumerate(losses.keys()):
            print(str(key) + ': ' + str(torch.mean(torch.FloatTensor(losses[key]))))


        '''Remain time calculation'''
        time_list.append(time.time()-start-sum(time_list))
        time_per_epoch = sum(time_list[1:])/(len(time_list)-1)
        time_remain = (args.num_epoch-(i+1))*time_per_epoch
        hour = int(time_remain/3600)
        delta = time_remain-hour*3600
        min = int(delta/60)
        delta = delta-min*60
        sec = int(delta)
        print("{}h {}m {}s Left".format(hour, min, sec))
        solver.save_frcnn()


def test():
    solver = Solver(args)
    solver.load_frcnn()
    solver.load_rpn()
    solver.test()


if __name__=='__main__':
    if args.train_rpn:
        train_rpn()
    elif args.train_frcnn:
        train_frcnn()
    elif args.test:
        test()
    else:
        raise Exception("Specify wheter to train or test module")

