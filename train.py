import csv
import time

import torch
import time
import torch.nn.functional as F
import torch.optim
import torch.utils.data
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from tqdm import tqdm

from data.data_entry import select_train_loader, select_eval_loader
from data.prepare_data import generate_dataloader
from model.model_entry import select_model
from options import prepare_train_args
from utils.logger import Logger
from utils.torch_utils import load_match_dict, AverageMeter, accuracy
from loss import MMDLoss


class Trainer:
    def __init__(self):
        args = prepare_train_args()
        self.args = args
        torch.manual_seed(args.seed)
        self.logger = Logger(args)

        # process data and prepare dataloaders
        self.train_loader_source, self.train_loader_target, self.val_loader_target, self.val_loader_source = generate_dataloader(
            args)

        self.model = select_model(args)
        if args.resume != '':
            print("=> using pre-trained weights.")
            if args.load_not_strict:
                load_match_dict(self.model, args.load_model_path)
            else:
                self.model.load_state_dict(torch.load(args.load_model_path).state_dict())

        self.cs_1 = Variable(torch.cuda.FloatTensor(args.num_classes, self.model.feat1_dim).fill_(0))
        self.ct_1 = Variable(torch.cuda.FloatTensor(args.num_classes, self.model.feat1_dim).fill_(0))
        self.cs_2 = Variable(torch.cuda.FloatTensor(args.num_classes, self.model.feat2_dim).fill_(0))
        self.ct_2 = Variable(torch.cuda.FloatTensor(args.num_classes, self.model.feat2_dim).fill_(0))

        self.optimizer = torch.optim.Adam(self.model.parameters(), self.args.lr,
                                          betas=(self.args.momentum, self.args.beta),
                                          weight_decay=self.args.weight_decay)

    def train(self):
        best_acc = 0
        for epoch in range(self.args.epochs):
            # train for one epoch
            self.train_per_epoch(epoch)
            val_acc = self.val_per_epoch(epoch)
            self.logger.save_curves(epoch)
            self.logger.save_check_point(self.model, val_acc, epoch)
            if val_acc > best_acc:
                best_acc = val_acc
        print("Best val acc is {:.2f}\n".format(best_acc))

    def train_per_epoch(self, epoch):
        top1 = AverageMeter()
        losses = AverageMeter()
        # switch to train mode
        self.model.train()

        train_loader_source_batch = enumerate(self.train_loader_source)
        train_loader_target_batch = enumerate(self.train_loader_target)
        batch_number = count_epoch_on_large_dataset(self.train_loader_target, self.train_loader_source)

        for itern in tqdm(range(batch_number)):
            pred_src, pred_cs_1, pred_ct_1, pred_cs_2, pred_ct_2, target_src_var = self.step(train_loader_source_batch,
                                                                                             train_loader_target_batch)

            prec1 = accuracy(pred_src.data, target_src_var, topk=(1,))[0]
            top1.update(prec1.item(), pred_src.size(0))

            # compute loss
            # metrics = self.compute_metrics(pred, label, is_train=True)

            # get the item for backward
            cs_target_var = Variable(torch.arange(0, self.args.num_classes).cuda(non_blocking=True))
            ct_target_var = Variable(torch.arange(0, self.args.num_classes).cuda(non_blocking=True))
            loss1 = self.compute_loss(pred_cs_1, cs_target_var, 'ce') + self.compute_loss(pred_ct_1, ct_target_var,
                                                                                          'ce')
            loss2 = self.compute_loss(pred_cs_2, cs_target_var, 'ce') + self.compute_loss(pred_ct_2, ct_target_var,
                                                                                          'ce')
            loss3 = self.compute_loss(pred_src, target_src_var, 'ce')
            loss4_1 = self.compute_loss(self.cs_1, self.ct_1, 'mmd')
            loss4_2 = self.compute_loss(self.cs_2, self.ct_2, 'mmd')
            loss = self.args.alpha1 * loss1 + self.args.alpha2 * loss2 + loss3 + self.args.alpha3 * loss4_1 + self.args.alpha4 * loss4_2
            losses.update(loss.item(), target_src_var.size(0))
            # compute gradient and do Adam step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # logger record
            metrics = {
                'train/' + 'prec1': prec1,
                'train/' + 'loss1': loss1,
                'train/' + 'loss2': loss2,
                'train/' + 'loss3': loss3,
                'train/' + 'loss4_1': loss4_1,
                'train/' + 'loss4_2': loss4_2,
                'train/' + 'losses': losses.val,
            }
            for key in metrics.keys():
                self.logger.record_scalar(key, metrics[key])

            # only save img at first step
            # if itern == len(self.train_loader) - 1:
            #     self.logger.save_imgs(self.gen_imgs_to_write(img, pred, label, True), epoch)

            # monitor training progress
            # if itern % self.args.print_freq == 0:
            #     print('\nTrain: Epoch {} batch {} Loss {} Acc {:.3f}'.format(epoch, itern, loss.avg, top1.avg))
        print('Train: Epoch {},  Loss {:.3f}, Acc {:.3f}'.format(epoch, losses.avg, top1.avg))
        return top1.avg

    def val_per_epoch(self, epoch):

        losses = AverageMeter()
        top1 = AverageMeter()

        # switch to evaluation mode
        self.model.eval()
        for i, (data, label) in tqdm(enumerate(self.train_loader_target)):
            target = label.cuda(non_blocking=True)
            input_var = Variable(data.cuda())
            target_var = Variable(target)

            # compute output
            with torch.no_grad():
                output = self.model(input_var)[-1]
                loss = self.compute_loss(output, target_var, loss_fuc='ce')

            # measure accuracy and record loss
            prec1 = accuracy(output.data, target, topk=(1,))[0]
            top1.update(prec1.item(), data.size(0))
            losses.update(loss.item(), data.size(0))
            '''
            if i % self.args.print_freq == 0:
                print('\nEvaluate on target - [{0}][{1}/{2}]\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'
                      .format(epoch, i, len(self.train_loader_target), loss=losses, top1=top1))
            '''
            # logger record
            metrics = {
                'val/' + 'prec1': prec1,
                'val/' + 'losses': losses.val,
            }
            for key in metrics.keys():
                self.logger.record_scalar(key, metrics[key])

        print(' * Evaluate on target - prec@1: {top1.avg:.3f}'.format(top1=top1))
        return top1.avg

    def step(self, train_loader_src_batch, train_loader_tar_batch):
        # prepare data for model forward and backward
        try:
            (input_source, target_source) = train_loader_src_batch.__next__()[1]
        except StopIteration:
            train_loader_src_batch = enumerate(self.train_loader_source)
            (input_source, target_source) = train_loader_src_batch.__next__()[1]

        try:
            (input_target, target_target_not_use) = train_loader_tar_batch.__next__()[1]
        except StopIteration:
            train_loader_tar_batch = enumerate(self.train_loader_target)
            (input_target, target_target_not_use) = train_loader_tar_batch.__next__()[1]

        target_source = target_source.cuda(non_blocking=True)
        # warp input
        # TODO 加 input_data.cuda()
        input_source_var = Variable(input_source.cuda())
        target_source_var = Variable(target_source)
        input_target_var = Variable(input_target.cuda())

        # cs_target_var = Variable(torch.arange(0, self.args.num_classes).cuda(non_blocking=True))
        # ct_target_var = Variable(torch.arange(0, self.args.num_classes).cuda(non_blocking=True))

        # model forward for source/target data
        feat1_s, feat2_s, pred_s = self.model(input_source_var)
        feat1_t, feat2_t, pred_t = self.model(input_target_var)
        prob_t = F.softmax(pred_t - pred_t.max(1, True)[0], dim=1)
        idx_max_prob = prob_t.topk(1, 1, True, True)[-1]  # 实质上得到tar的预测class
        # 这里实质是将所有结果统一变成负的，再进行softmax操作  ?????
        '''
        x.topk(k, dim=None, largest=True, sorted=True)
        k 表示保留k个值
        dim 表示维度方向 dim=0是指在行维度取进行最大，也就是寻找每一列的最大值；dim=1是指在列维度进行取最大，也就是寻找每一行的最大值
        largest=True意味着选取最大的，sorted=True是指将返回结果排序
        返回一个tuple，一个是values,另一个是indices
        '''
        # compute source and target centroids on respective batches at the current iteration
        cs_1_temp = Variable(torch.cuda.FloatTensor(self.cs_1.size()).fill_(0))  # shape(65, 2048)
        cs_count = torch.cuda.FloatTensor(self.args.num_classes, 1).fill_(0)  # shape(65, 1)
        ct_1_temp = Variable(torch.cuda.FloatTensor(self.ct_1.size()).fill_(0))  # shape(65, 2048)
        ct_count = torch.cuda.FloatTensor(self.args.num_classes, 1).fill_(0)  # shape(65, 1)

        cs_2_temp = Variable(torch.cuda.FloatTensor(self.cs_2.size()).fill_(0))  # shape(65, 512)
        ct_2_temp = Variable(torch.cuda.FloatTensor(self.ct_2.size()).fill_(0))  # shape(65, 512)

        for i in range(self.args.batch_size):
            # 循环一个batch的次数，将属于同一个类别的所有图片的特征进行叠加
            cs_1_temp[target_source[i]] += feat1_s[i]  # cs_1_temp 将特征按照类别顺序1-65进行存放
            cs_count[target_source[i]] += 1  # 计算源域特征，在每个类别中存放的个数，即一个batch中，同一类别的图像个数
            cs_2_temp[target_source[i]] += feat2_s[i]
            ct_1_temp[idx_max_prob[i]] += feat1_t[i]  # 根据目标域的伪标签存放目标域的特征，按照类别顺序1-65进行存放
            ct_count[idx_max_prob[i]] += 1  # 计算一个batch中目标域属于同一类别图片的个数（后续可以求平均值）
            ct_2_temp[idx_max_prob[i]] += feat2_t[i]

        # exponential moving average centroids
        cs_1 = Variable(self.cs_1.data.clone())  # 这里每次迭代，都会在上一个epoch的基础上进行变化
        ct_1 = Variable(self.ct_1.data.clone())
        cs_2 = Variable(self.cs_2.data.clone())
        ct_2 = Variable(self.ct_2.data.clone())  # shape(65,512)
        # 统计cs_1.data != 0 的个数
        '''
        cs_1.data  shape=(65, 512)
        (cs_1.data != 0).sum(1, keepdim=True).shape = (65,1)
        ((cs_1.data != 0).sum(1, keepdim=True) != 0).shape = (65, 1)
        .float()将布尔型转换为浮点型
        '''
        mask_s = ((cs_1.data != 0).sum(1, keepdim=True) != 0).float() * self.args.remain
        mask_t = ((ct_1.data != 0).sum(1, keepdim=True) != 0).float() * self.args.remain
        mask_s[cs_count == 0] = 1.0
        mask_t[ct_count == 0] = 1.0
        cs_count[cs_count == 0] = self.args.eps  # 对于batch中某个类别没有输入图片的，设置一个下溢系数
        ct_count[ct_count == 0] = self.args.eps
        self.cs_1 = mask_s * cs_1 + (1 - mask_s) * (cs_1_temp / cs_count)  # mask 的范围0-1，大多数0.7
        self.ct_1 = mask_t * ct_1 + (1 - mask_t) * (ct_1_temp / ct_count)
        self.cs_2 = mask_s * cs_2 + (1 - mask_s) * (cs_2_temp / cs_count)
        self.ct_2 = mask_t * ct_2 + (1 - mask_t) * (ct_2_temp / ct_count)

        # centroid forward
        pred_cs_1 = self.model.fc2(self.model.fc1(cs_1))
        pred_ct_1 = self.model.fc2(self.model.fc1(ct_1))
        pred_cs_2 = self.model.fc2(cs_2)
        pred_ct_2 = self.model.fc2(ct_2)

        return pred_s, pred_cs_1, pred_ct_1, pred_cs_2, pred_ct_2, target_source_var

    def compute_metrics(self, pred, gt, is_train):
        # you can call functions in metrics.py
        l1 = (pred - gt).abs().mean()
        prefix = 'train/' if is_train else 'val/'
        metrics = {
            prefix + 'l1': l1
        }
        return metrics

    def gen_imgs_to_write(self, img, pred, label, is_train):
        # override this method according to your visualization
        prefix = 'train/' if is_train else 'val/'
        return {
            prefix + 'img': img[0],
            prefix + 'pred': pred[0],
            prefix + 'label': label[0]
        }

    def compute_loss(self, pred, gt, loss_fuc):
        if loss_fuc == 'l1':
            loss = (pred - gt).abs().mean()
        elif loss_fuc == 'ce':
            loss = torch.nn.functional.cross_entropy(pred, gt)
        elif loss_fuc == 'mmd':
            # mmd = MMDLoss()
            loss = mmd_rbf(source=pred, target=gt)
        else:
            loss = torch.nn.functional.mse_loss(pred, gt)
        return loss


def count_epoch_on_large_dataset(train_loader_target, train_loader_source):
    batch_number_t = len(train_loader_target)
    batch_number = batch_number_t
    batch_number_s = len(train_loader_source)
    if batch_number_s > batch_number_t:
        batch_number = batch_number_s

    return batch_number


def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    """
    将源域数据和目标域数据转化为核矩阵，即上文中的K
    Params:
	    source: 源域数据（n * len(x))
	    target: 目标域数据（m * len(y))
	    kernel_mul:
	    kernel_num: 取不同高斯核的数量
	    fix_sigma: 不同高斯核的sigma值
	Return:
		sum(kernel_val): 多个核矩阵之和
    """
    n_samples = int(source.size()[0]) + int(target.size()[0])  # 求矩阵的行数，一般source和target的尺度是一样的，这样便于计算
    total = torch.cat([source, target], dim=0)  # 将source,target按列方向合并
    # 将total复制（n+m）份
    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    # 将total的每一行都复制成（n+m）行，即每个数据都扩展成（n+m）份
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    # 求任意两个数据之间的和，得到的矩阵中坐标（i,j）代表total中第i行数据和第j行数据之间的l2 distance(i==j时为0）
    L2_distance = ((total0 - total1) ** 2).sum(2)
    # 调整高斯核函数的sigma值
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
    # 以fix_sigma为中值，以kernel_mul为倍数取kernel_num个bandwidth值（比如fix_sigma为1时，得到[0.25,0.5,1,2,4]
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
    # 高斯核函数的数学表达式
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    # 得到最终的核矩阵
    return sum(kernel_val)  # /len(kernel_val)


def mmd_rbf(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    """
    计算源域数据和目标域数据的MMD距离
    Params:
	    source: 源域数据（n * len(x))
	    target: 目标域数据（m * len(y))
	    kernel_mul:
	    kernel_num: 取不同高斯核的数量
	    fix_sigma: 不同高斯核的sigma值
	Return:
		loss: MMD loss
    """
    batch_size = int(source.size()[0])  # 一般默认为源域和目标域的batchsize相同
    kernels = guassian_kernel(source, target,
                              kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    # 根据式（3）将核矩阵分成4部分
    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    YX = kernels[batch_size:, :batch_size]
    loss = torch.mean(XX + YY - XY - YX)
    return loss  # 因为一般都是n==m，所以L矩阵一般不加入计算


def main():
    cudnn.enabled = True
    cudnn.benchmark = True

    trainer = Trainer()
    trainer.train()


if __name__ == '__main__':
    main()
