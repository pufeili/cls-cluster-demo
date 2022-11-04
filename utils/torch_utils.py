import torch


def load_match_dict(model, model_path):
    # model: single gpu model, please load dict before warp with nn.DataParallel
    pretrain_dict = torch.load(model_path)
    model_dict = model.state_dict()
    # the pretrain dict may be multi gpus, cleaning
    pretrain_dict = {k.replace('.module', ''): v for k, v in pretrain_dict.items()}
    # 1. filter out unnecessary keys
    pretrain_dict = {k: v for k, v in pretrain_dict.items() if
                     k in model_dict and v.shape == model_dict[k].shape}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrain_dict)
    # 3. load the new state dict
    model.load_state_dict(model_dict)


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
        # n 表示所有sum, count 的值都计算n遍, n的设置不会影响avg, val
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    pred = torch.tensor([[0.5, 0.6, 0.8, 0.7, 0.4],
                         [0.2, 0.6, 0.4, 0.3, 0.5]])
    label = torch.tensor([[1],
                          [1]])
    acc = accuracy(pred, label)
    print(acc)

    res = AverageMeter()
    res2 = AverageMeter()
    a = torch.tensor([0.2, 0.4, 0.6, 0.8, 1.0])
    for i in range(5):
        res.update(a[i], 2)
        res2.update(a[i])
        print(res.val, '\t', res.sum, '\t', res.count, '\t', res.avg)
        print(res2.val, '\t', res2.sum, '\t', res2.count, '\t', res2.avg)
        print("=============")
    run_code = 0
