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
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--dataset', type=str, default='mnist',
                    help='Define which dataset will be adopted (default: mnist) choices=> mnist, cifar10')
parser.add_argument('--mode', type=str, default='train', help='mode of run')
parser.add_argument('--inlier', type=str, default='0,1,2,3,4,5,6,7,8,9', help='Define inlier class')
# parser.add_argument('--num_step1', type=int, default=1, help='step 1 train number')
# parser.add_argument('--num_step2_ae', type=int, default=1, help='step 2 ae train number')
# parser.add_argument('--num_step2_gan', type=int, default=1, help='step 2 gan train number')
# parser.add_argument('--num_step3', type=int, default=1, help='step 3 train number')
# parser.add_argument('--num_step4', type=int, default=1, help='step 4 train number')
# parser.add_argument('--num_step6', type=int, default=1, help='step 4 train number')
parser.add_argument('--batch_size', type=int, default=4, help='batch size')
parser.add_argument('--num_epoch', type=int, default=50, help='num_epoch')
# parser.add_argument('--threshold', type=float, default=0.001, help='threshold')
parser.add_argument('--train_rpn', action='store_true', default=False, help='train model')
parser.add_argument('--train_frcnn', action='store_true', default=False, help='train model')
# parser.add_argument('--train_post', action='store_true', default=False, help='train model')
parser.add_argument('--test', action='store_true', default=False, help='test model')
parser.add_argument('--output_dir', type=str, default='outputs1', help='result histogram, generated images directory')
parser.add_argument('--restoration_lr', type=float, default=0.0002, help='restoration adam learning rate')
parser.add_argument('--mask_generation_lr', type=float, default=0.0002, help='mask generation adam learning rate')
parser.add_argument('--mask_zeros_lr', type=float, default=0.0002, help='mask zeros adam learning rate')
parser.add_argument('--autoencoder', action='store_true', default=False, help='whether to use autoencoder')
parser.add_argument('--mask_model', type=str, default=None, help='which model will be used to generate mask')
parser.add_argument('--denoising_model', type=str, default=None, help='which model will be used to denoise masked images')

args = parser.parse_args()

args.cuda = not args.no_cuda and torch.cuda.is_available()


'''To reproduce score => To make deterministic'''
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
torch.backends.cudnn.deterministic=True

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
        # solver.save_frcnn()

def test():
    solver = Solver(args)
    solver.load_model(path=args.dataset)
    test_result = solver.test(mode='inlier')
    for _, key in enumerate(test_result.keys()):
        print(str(key) + ': ' + str(torch.mean(torch.FloatTensor(test_result[key]))))
    test_result = solver.test(mode='outlier')
    for _, key in enumerate(test_result.keys()):
        print(str(key) + ': ' + str(torch.mean(torch.FloatTensor(test_result[key]))))

if __name__=='__main__':
    if args.train_rpn:
        train_rpn()
    elif args.train_frcnn:
        train_frcnn()
    elif args.test:
        test()
    else:
        raise Exception("Specify wheter to train or test module")

