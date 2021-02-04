"""
Examples:
python main.py -a bagnetscattering --scattering-J 4 --scattering-order2 --layer-width 1024 --n-iterations 4 -j 4 /d1/dataset/2012
python main.py -a scatteringlinear -j 10 --scattering-J 4 -j 10 --scattering-oversampling 1 /ssd/dataset/2012/ --epochs 100
"""
import argparse
import os
import random
import shutil
import time
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import pdb
from kymatio import Scattering2D
from models import BagNetScattering, ScatteringLinear
from utils import print_and_write

# import matplotlib.pyplot as plt
# from tensorboardX import SummaryWriter

model_names = [
        'bagnetscattering',
        'scatteringlinear',
]

np.set_printoptions(precision=3, suppress=True)

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

# args of the original script
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--arch', '-a', choices=model_names,  help='model architecture')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--inspect-model', action='store_true',
                    help='activate pdb to inspect the model')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of distributed processes')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='gloo', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')

#args for logs and save
parser.add_argument('--logs-dir', default='./training_logs', type=str, help='directory for the logs')
parser.add_argument('--checkpoint-savedir', default='./checkpoints', type=str, help='directory to save checkpoints')
parser.add_argument('--print-freq', '-p', default=200, type=int,
                    metavar='N', help='print frequency (default: 100)')

# args for training
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--lr-thetas', default=0., type=float,
                    metavar='LR theta', help='initial learning rate for the thetas parameters of BagScatNetTiedLista')
parser.add_argument('--learning-rate-adjust-frequency', default=30, type=int,
                    help='number of epoch after which learning rate is decayed by 10 (default: 30)')
parser.add_argument('--val-resize', default=256, type=int,
        help='resize before center crop in validation (default: 256)')
parser.add_argument('--train-cropsize', default=224, type=int,
        help='cropsize for the train (default: 224)')
parser.add_argument('--nb-classes', default=1000, type=int,
        help='number of classes randomly chosen used for training and validation (default: 1000 = whole train/val dataset)')
parser.add_argument('--class-indices', default=None, help='numpy array of indices used in case nb-classes < 1000')
parser.add_argument('--random-seed', default=None, type=int, help='random seed used')
parser.add_argument('--new-optimizer', help='don\'t load the optimizer when resuming',
            action='store_true')


# args for scattering transform
parser.add_argument('--scattering-J', default=4, type=int,
        help='j value (= maximum scale) for the scattering transform')
parser.add_argument('--scattering-order2', help='Compute order2 scattering coefficients',
            action='store_true')
parser.add_argument('--scattering-oversampling', default=0, type=int, help='Oversampling factor for scattering')
parser.add_argument('--scattering-fft', help='adds ft mod to scattering', action='store_true')

# args for the architecture
parser.add_argument('--conv1', default=3, type=int, help='kernel size of the first conv layer (default: 3)')
parser.add_argument('--stride1', default=2, type=int, help='stride of the first conv layer of  (default: 2)')
parser.add_argument('--layer-width', default=1024, type=int, help='width of the 1x1 layers')
parser.add_argument('--layer-width-2', default=1024, type=int, help='width of the second 1x1 layers')
parser.add_argument('--proj-dim', default=512, type=int,
                    help='dimension of the initial projection')
parser.add_argument('--n-iterations', default=4, type=int, help='number of iterations for ISTA like algos')
parser.add_argument('--n-iterations-2', default=4, type=int, help='2nd number of iterations for Bi-level ISTA like algos')
parser.add_argument('--theta-init', default=1., type=float,
        help='theta init for models with both fixed and flexible thetas')
parser.add_argument('--bn1', action='store_true', help='add batch norm with no affine after proj')
parser.add_argument('--one-over-L', action='store_true', help='use one over L in Ista models')


# args for auxilary losses
parser.add_argument('--reconstruction-lambda', default=0., type=float, help='reconstruction loss lambda')
parser.add_argument('--reclassification-lambda', default=0., type=float, help='reclassification loss lambda')


# args for sparse coding verification
parser.add_argument('--compare-with-fista', action='store_true', help='compare l1 convergence with fista')


best_acc1 = 0
best_acc5 = 0
best_epoch_acc1 = 0
best_epoch_acc5 = 0


def main():
    global args, best_acc1, best_acc5, best_epoch_acc1, best_epoch_acc5
    args = parser.parse_args()



    logs_dir = args.logs_dir
    if not os.path.exists(logs_dir):
            os.makedirs(logs_dir)

    checkpoint_savedir = args.checkpoint_savedir
    if not os.path.exists(checkpoint_savedir):
            os.makedirs(checkpoint_savedir)

    logfile = os.path.join(logs_dir, 'training_{}_b_{}_lrfreq_{}.log'.format(
        args.arch, args.batch_size, args.learning_rate_adjust_frequency))

    summaryfile = os.path.join(logs_dir, 'summary_file.txt')


    checkpoint_savefile = os.path.join(checkpoint_savedir, '{}_batchsize_{}_lrfreq_{}_traincropsize_{}_fft{}.pth.tar'.format(
        args.arch, args.batch_size, args.learning_rate_adjust_frequency, args.train_cropsize, args.scattering_fft))
    best_checkpoint_savefile = os.path.join(checkpoint_savedir, '{}_batchsize_{}_lrfreq_{}_traincropsize_{}_best_fft{}.pth.tar'.format(
        args.arch, args.batch_size, args.learning_rate_adjust_frequency, args.train_cropsize, args.scattering_fft))


    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    args.distributed = args.world_size > 1

    if args.distributed:
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size)

    dictionary = None

    max_order = 2 if args.scattering_order2 else 1
    print_and_write("=> Scattering transform order {}, J {}".format(max_order, args.scattering_J), logfile=logfile)
    pre_transformation = torch.nn.DataParallel(Scattering2D(J=args.scattering_J, shape=(args.train_cropsize, args.train_cropsize),
                                      max_order=max_order, oversampling=args.scattering_oversampling)).cuda()
    if args.arch == 'bagnetscattering':
        arch_log = "=> creating model 'bagnetscattering' oversampling {} width {}, {} iterations".format(args.scattering_oversampling,  args.layer_width, args.n_iterations)
        model = BagNetScattering(J=args.scattering_J, N=args.train_cropsize, layer_width=args.layer_width, order2=args.scattering_order2,
                                    n_iterations=args.n_iterations, first_layer_kernel_size=args.conv1,
                                    skip_stride=args.stride1)
    if args.arch == 'scatteringlinear':
        arch_log = "=> creating model 'scatteringlinear' oversampling {} fft {}".format(args.scattering_oversampling, args.scattering_fft)
        J = args.scattering_J
        oversampling = args.scattering_oversampling
        if oversampling == 0:
            n_space = int(224 // 2**J)
        elif oversampling == 1:
            n_space = int(224 // 2**(J-1)) + 2
        model = ScatteringLinear(n_space=n_space, J=args.scattering_J, order2=args.scattering_order2, use_fft=args.scattering_fft)

    else:
        print("=> unknown model '{}'".format(args.arch))
        return

    print_and_write(arch_log, logfile=logfile)

    print_and_write("=> learning rate {}, lr thetas {}, {} epochs, decay frequency {}".format(args.lr, args.lr_thetas, args.epochs, args.learning_rate_adjust_frequency), logfile=logfile)

    model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion)
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)


    # define the optimizer
    if args.arch == 'bagscatnetlistaproj':
        params_list = [
            {'params': model.module.proj.parameters()},
            {'params': model.module.skip.parameters()},
            {'params': model.module.conv.parameters()},
            {'params': model.module.fc.parameters()},
            {'params': model.module.bn.parameters()},
            {'params': model.module.bn0.parameters()},
            {'params': model.module.thetas, 'lr': 0., 'name': 'thetas', 'weight_decay': 0},
        ]
    elif args.arch == 'bagscatnetlista2proj':
        params_list = [
            {'params': model.module.proj.parameters()},
            {'params': model.module.WT.parameters()},
            {'params': model.module.D.parameters()},
            {'params': model.module.fc.parameters()},
            {'params': model.module.bn.parameters()},
            {'params': model.module.bn0.parameters()},
            {'params': model.module.thetas, 'lr': 0., 'name': 'thetas', 'weight_decay': 0},
        ]
    if args.arch == 'bagscatnetistaproj':
        params_list = [
            {'params': model.module.bn0.parameters()},
            {'params': model.module.proj.parameters()},
            {'params': model.module.dictionary},
            {'params': model.module.fc.parameters()},
            {'params': model.module.bn.parameters()},
            {'params': model.module.thetas, 'lr': 0., 'name': 'thetas', 'weight_decay': 0},
        ]
    if args.arch == 'bagscatnetslistaproj':
        params_list = [
            {'params': model.module.bn0.parameters()},
            {'params': model.module.proj.parameters()},
            {'params': model.module.dictionary},
            {'params': model.module.fc.parameters()},
            {'params': model.module.bn.parameters()},
            {'params': model.module.theta, 'lr': 0., 'name': 'thetas', 'weight_decay': 0},
            {'params': model.module.gammas, 'lr': 0., 'name': 'thetas', 'weight_decay': 0},
        ]
    else:
        params_list = model.parameters()

    optimizer = torch.optim.SGD(params_list, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)


    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            if args.start_epoch == 0:
                args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            model.load_state_dict(checkpoint['state_dict'])
            if not args.new_optimizer:
                optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    if args.inspect_model:
        pdb.set_trace()

    cudnn.benchmark = True

    # Data loading code

    if args.nb_classes < 1000:
        if args.random_seed is not None:
            torch.manual_seed(args.random_seed)

        train_indices = list(np.load('utils_sampling/imagenet_train_class_indices.npy'))
        val_indices = list(np.load('utils_sampling/imagenet_val_class_indices.npy'))
        classes_names = torch.load('utils_sampling/labels_dict')
        if args.class_indices is not None:
            class_indices = torch.load(args.class_indices)
        else:
            perm = torch.randperm(1000)
            class_indices = perm[:args.nb_classes].tolist()
        train_indices_full = [x for i in range(len(class_indices)) for x in range(train_indices[class_indices[i]],
                                                                                    train_indices[class_indices[i]+1])]
        val_indices_full = [x for i in range(len(class_indices)) for x in range(val_indices[class_indices[i]],
                                                                                    val_indices[class_indices[i]+1])]
        classes_indices_file = os.path.join(logs_dir, 'classes_indices_selected')
        classes_names_file = os.path.join(logs_dir, 'classes_names_selected')
        classes_file_txt = os.path.join(logs_dir, 'classes_selected.txt')
        f = open(classes_file_txt, 'w')
        selected_classes_names = [classes_names[i] for i in class_indices]
        torch.save(class_indices, classes_indices_file)
        torch.save(selected_classes_names, classes_names_file)
        print('\nSelected {} classes indices:  {}\n'.format(args.nb_classes, class_indices))
        print('Selected {} classes names:  {}\n'.format(args.nb_classes, selected_classes_names))
        if args.random_seed is not None:
            print('Random seed used {}\n'.format(args.random_seed))


        print('Selected {} classes indices:  {}\n'.format(args.nb_classes, class_indices), file=f)
        print('Selected {} classes names:  {}\n'.format(args.nb_classes, selected_classes_names), file=f)
        if args.random_seed is not None:
            print('Random seed used {}\n'.format(args.random_seed), file=f)
        f.close()


    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(args.train_cropsize),
            # transforms.Resize(args.val_resize),
            # transforms.RandomCrop(args.train_cropsize),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    if args.nb_classes < 1000:
        train_dataset = torch.utils.data.Subset(train_dataset, train_indices_full)


    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_dataset = datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(args.val_resize),
            transforms.CenterCrop(args.train_cropsize),
            transforms.ToTensor(),
            normalize,
        ]))

    if args.nb_classes < 1000:
        val_dataset = torch.utils.data.Subset(val_dataset, val_indices_full)

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                             num_workers=args.workers, pin_memory=True)


    if args.evaluate:
        validate(val_loader, model, criterion, pre_transformation=pre_transformation, compare_with_fista=args.compare_with_fista)
        return


    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        adjust_learning_rate(optimizer, epoch, adjust_frequency=args.learning_rate_adjust_frequency)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, pre_transformation=pre_transformation, logfile=logfile,
              dictionary=dictionary)


        # evaluate on validation set
        acc1, acc5 = validate(val_loader, model, criterion, epoch=epoch,
                              pre_transformation=pre_transformation, logfile=logfile)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
        if is_best:
            best_epoch_acc1 = epoch
        if acc5 > best_acc5:
            best_acc5 = acc5
            best_epoch_acc5 = epoch
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_acc1': best_acc1,
            'optimizer': optimizer.state_dict(),
        }, is_best, checkpoint_filename=checkpoint_savefile, best_checkpoint_filename=best_checkpoint_savefile)


    sum_up_log = "Best top 1 accuracy {:.2f} at epoch {}, best top 5 accuracy {:.2f} at epoch {}".format(
          best_acc1, best_epoch_acc1, best_acc5, best_epoch_acc5)
    print_and_write(sum_up_log, logfile=logfile)
    print_and_write(sum_up_log, logfile=summaryfile)


def train(train_loader, model, criterion, optimizer, epoch, pre_transformation=None, logfile=None, dictionary=None, writer=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    pre_transformation_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end = time.time()

    model.train()

    # if epoch == 0:
        # print_and_write('warm up for batch norms', logfile=logfile)
        # for i, (input, target) in enumerate(train_loader):
            # input = input.cuda()

            # if pre_transformation is not None:
                # with torch.no_grad():
                    # input = pre_transformation(input)
                    # output = model(input)
                    # if i > 200:
                        # break
    # end = time.time()

    for i, (input, target) in enumerate(train_loader):
        data_time.update(time.time() - end)

        input = input.cuda()

        # pre-processing
        if pre_transformation is not None:
            now = time.time()
            with torch.no_grad():
                input = pre_transformation(input)
            pre_transformation_time.update(time.time() - now)


        target = target.cuda(args.gpu, non_blocking=True)
        output = model(input)

        loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))


        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print_and_write('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.1f} ({batch_time.avg:.1f})\t'
                  'Data {data_time.val:.1f} ({data_time.avg:.1f})\t'
                  'preproc. {pre_transformation_time.val:.1f}({pre_transformation_time.avg:.1f})\t'
                  'Tot. Loss {loss.val:.2f} ({loss.avg:.2f})\t'
                  'Acc@1 {top1.val:.1f} ({top1.avg:.1f})\t'
                  'Acc@5 {top5.val:.1f} ({top5.avg:.1f})\t'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, pre_transformation_time=pre_transformation_time,
                   loss=losses, top1=top1, top5=top5), logfile=logfile)



def validate(val_loader, model, criterion, epoch=None, writer=None, pre_transformation=None, logfile=None, summaryfile=None,
             force_avgpool=False, last_conv_layer_PCA=None, unfold=None, multi_scale=False, whitening=None, compare_with_fista=False):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    pre_transformation_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):

            input = input.cuda()

            if pre_transformation is not None:
                input = pre_transformation(input)

            target = target.cuda(args.gpu, non_blocking=True)

            output = model(input)

            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))



            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print_and_write('Validation Epoch: [{0}][{1}/{2}]\t'
                                'Time {batch_time.val:.1f} ({batch_time.avg:.1f})\t'
                                'Data {data_time.val:.1f} ({data_time.avg:.1f})\t'
                                'preproc. {pre_transformation_time.val:.1f}({pre_transformation_time.avg:.1f})\t'
                                'Tot. Loss {loss.val:.2f} ({loss.avg:.2f})\t'
                                'Acc@1 {top1.val:.1f} ({top1.avg:.1f})\t'
                                'Acc@5 {top5.val:.1f} ({top5.avg:.1f})\t'.format(
                    epoch, i, len(val_loader), batch_time=batch_time,
                    data_time=data_time, pre_transformation_time=pre_transformation_time,
                    loss=losses, top1=top1, top5=top5), logfile=logfile)

        print_and_write('Validation Epoch {}, * Acc@1 {:.2f} Acc@5 {:.2f}'.format(epoch, top1.avg, top5.avg), logfile=logfile)


    return top1.avg, top5.avg


def save_checkpoint(state, is_best, checkpoint_filename='checkpoint.pth.tar',
                    best_checkpoint_filename='model_best.pth.tar'):
    torch.save(state, checkpoint_filename)
    if is_best:
        shutil.copyfile(checkpoint_filename, best_checkpoint_filename)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class AverageMeterTensor(object):
    """Computes and stores the average and current value"""
    def __init__(self, size):
        self.reset(size)

    def reset(self, size):
        self.avg = np.zeros(size)
        self.sum = np.zeros(size)
        self.count = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch, adjust_frequency=30, lr_decay_rate=0.1):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (lr_decay_rate ** (epoch // adjust_frequency))
    lr_thetas = args.lr_thetas * (lr_decay_rate ** (epoch // adjust_frequency))

    for param_group in optimizer.param_groups:
        if 'name' in param_group.keys() and param_group['name'] == 'thetas':
            param_group['lr'] = lr_thetas
        else:
            param_group['lr'] = lr




def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res



if __name__ == '__main__':
    main()
