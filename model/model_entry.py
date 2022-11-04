from model.base.fcn import CustomFcn
from model.best.fcn import DeepLabv3Fcn
from model.better.fcn import Resnet101Fcn
from model.sota.fcn import LightFcn
from model.base.resnet import resnet50  # DisClusterDA
import torch.nn as nn


def select_model(args):
    type2model = {
        'resnet50_fcn': CustomFcn(args),
        'resnet101_fcn': Resnet101Fcn(args),
        'deeplabv3_fcn': DeepLabv3Fcn(args),
        'resnet50': resnet50(args),
    }
    # 'mobilnetv3_fcn': LightFcn(args),
    model = type2model[args.model_type]

    # define multi-GPU
    if len(args.gpus) > 1:
        model = equip_multi_gpu(model, args)
        model = model.module
    elif len(args.gpus) == 1:
        model = model.cuda()
    return model


def equip_multi_gpu(model, args):
    model = nn.DataParallel(model, device_ids=args.gpus).cuda()
    return model


if __name__ == '__main__':

    run_code = 0
