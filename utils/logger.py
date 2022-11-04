from tensorboardX import SummaryWriter
import os
import torch


class Recoder:
    def __init__(self):
        self.metrics = {}

    def record(self, name, value):
        if name in self.metrics.keys():
            self.metrics[name].append(value)
        else:
            self.metrics[name] = [value]

    def summary(self):
        kvs = {}
        for key in self.metrics.keys():
            kvs[key] = sum(self.metrics[key]) / len(self.metrics[key])
            del self.metrics[key][:]
            self.metrics[key] = []
        return kvs


class Logger:
    def __init__(self, args):
        self.writer = SummaryWriter(args.model_dir)
        self.recoder = Recoder()
        self.model_dir = args.model_dir

        self.max_epoch = args.epochs

    def tensor2img(self, tensor):
        # implement according to your data, for example call viz.py
        return tensor.cpu().numpy()

    def record_scalar(self, name, value):
        self.recoder.record(name, value)

    def save_curves(self, epoch):
        kvs = self.recoder.summary()
        for key in kvs.keys():
            self.writer.add_scalar(key, kvs[key], epoch)
        return kvs

    def save_imgs(self, names2imgs, epoch):
        for name in names2imgs.keys():
            self.writer.add_image(name, self.tensor2img(names2imgs[name]), epoch)

    def save_check_point(self, model, val_acc, epoch=0, step=0):
        # model_name = '{epoch:02d}_{step:06d}.pth'.format(epoch=epoch, step=step)
        prec1 = val_acc
        best_prec1 = 0
        if prec1 > best_prec1:
            best_prec1 = prec1
            model_name = 'best_checkpoint.pth'
            path = os.path.join(self.model_dir, model_name)
            # don't save model, which depends on python path
            # save model state dict
            torch.save(model.state_dict(), path)
        if epoch == (self.max_epoch-1):
            model_name = 'final_checkpoint_{:03d}.pth'.format(epoch+1)
            path = os.path.join(self.model_dir, model_name)
            torch.save(model.state_dict(), path)


if __name__ == '__main__':
    a = [1, 3, 4, 6]
    print(a[-1])
    exit()

    from options import prepare_train_args
    args = prepare_train_args()
    print(args.model_dir)
    logs = Logger(args)

    a, b, c = 1, 1, 1
    for e in range(50):
        for i in range(20):
            metrics = {
                'train/' + 'loss1': a,
                'train/' + 'loss2': b,
                'train/' + 'loss3': c,
            }
            for key in metrics.keys():
                logs.record_scalar(key, metrics[key])
            a += 1
            b += 3
            c += 5
        logs.save_curves(e)
    run_code = 0
