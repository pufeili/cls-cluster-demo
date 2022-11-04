import argparse
import os


def parse_common_args(parser):
    parser.add_argument('--model_type', type=str, default='resnet50',
                        help='used in model_entry.py')  # default='base_model'
    parser.add_argument('--data_type', type=str, default='base_dataset', help='used in data_entry.py')
    parser.add_argument('--save_prefix', type=str, default='no_squeeze',
                        help='some comment for model or test result dir')
    parser.add_argument('--load_model_path', type=str,
                        default='checkpoints/amazon2dslr_a_1.0_b_1.0_c_1.0_d_1.0/best_checkpoint.pth',
                        help='model path for pretrain or test')  # checkpoints/base_model_pref/0.pth
    parser.add_argument('--load_not_strict', action='store_true', help='allow to load only common state dicts')

    parser.add_argument('--gpus', default=[0, ], nargs='+', type=int)
    parser.add_argument('--seed', type=int, default=3407)

    # datasets
    parser.add_argument('--data_path_source', type=str, default='./data/office31/',
                        help='root of the source dataset')
    parser.add_argument('--data_path_target_tr', type=str, default='./data/office31/',
                        help='root of the target dataset (for training)')
    parser.add_argument('--data_path_target_te', type=str, default='./data/office31/',
                        help='root of the target dataset (for test)')
    parser.add_argument('--src', type=str, default='amazon', help='source domain')
    parser.add_argument('--tar_tr', type=str, default='dslr', help='target domain (for training)')
    parser.add_argument('--tar_te', type=str, default='dslr', help='target domain (for test)')
    parser.add_argument('--num_classes', type=int, default=31, help='class number')

    parser.add_argument('--eps', type=float, default=1e-6, help='a small value to prevent underflow')
    parser.add_argument('--no_da', action='store_true', help='whether using data augmentation')
    parser.add_argument('--gray_tar_agree', action='store_true',
                        help='whether to enforce the consistency between RGB and gray images on the target domain')
    parser.add_argument('--aug_tar_agree', action='store_true',
                        help='whether to enforce the consistency between RGB and augmented images on the target domain')
    parser.add_argument('--workers', default=0, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--remain', default=0.7, type=float,
                        help='the remaining weight of centroid of last epoch at this epoch (a number in [0, 1))')
    parser.add_argument('--print_freq', default=20, type=int,
                        metavar='N', help='print frequency (default: 10)')
    # architecture
    parser.add_argument('--arch', type=str, default='resnet50', help='model name')
    parser.add_argument('--resume', type=str, default='', help='checkpoint path to resume (default: '')')
    parser.add_argument('--pretrained', default=True, action='store_true', help='whether using pretrained model')
    parser.add_argument('--num_neurons', type=int, default=128, help='number of neurons in the fc1 of a new model')
    return parser


def parse_train_args(parser):
    parser = parse_common_args(parser)
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum for sgd, alpha parameter for adam')
    parser.add_argument('--beta', default=0.999, type=float, metavar='M',
                        help='beta parameters for adam')
    parser.add_argument('--weight-decay', '--wd', default=0, type=float,
                        metavar='W', help='weight decay')
    parser.add_argument('--model_dir', type=str, default='', help='leave blank, auto generated')
    parser.add_argument('--train_list', type=str, default='/data/dataset1/list/base/train.txt')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=3)

    # loss coefficient
    parser.add_argument('--alpha1', type=float, default=1.0)
    parser.add_argument('--alpha2', type=float, default=1.0)
    parser.add_argument('--alpha3', type=float, default=1.0)
    parser.add_argument('--alpha4', type=float, default=1.0)
    return parser


def parse_test_args(parser):
    parser = parse_common_args(parser)
    parser.add_argument('--save_viz', action='store_true', help='save viz result in eval or not')
    parser.add_argument('--result_dir', type=str, default='', help='leave blank, auto generated')

    parser.add_argument('--batch_size', type=int, default=64)
    return parser


def get_train_args():
    parser = argparse.ArgumentParser()
    parser = parse_train_args(parser)
    args = parser.parse_args()
    return args


def get_test_args():
    parser = argparse.ArgumentParser()
    parser = parse_test_args(parser)
    args = parser.parse_args()
    return args


def get_train_model_dir(args):
    model_dir = os.path.join('checkpoints', args.src + '2' + args.tar_tr + '_a_' + str(args.alpha1) + '_b_' + str(
        args.alpha2) + '_c_' + str(args.alpha3) + '_d_' + str(args.alpha4))
    if not os.path.exists(model_dir):
        os.system('mkdir -p ' + model_dir)
    args.model_dir = model_dir


def get_test_result_dir(args):
    # ext = os.path.basename(args.load_model_path).split('.')[-1]  # 返回path最后的文件名
    model_dir = args.load_model_path.replace('.pth', '')
    # os.path.dirname 去掉文件名，返回目录 val_info = val

    # val_info = os.path.basename(os.path.dirname(args.val_list)) + '_' \
    #            + os.path.basename(args.val_list.replace('.txt', ''))
    val_info = args.src + '2' + args.tar_te
    # result_dir = os.path.join(model_dir, val_info + '_' + args.save_prefix)
    result_dir = model_dir
    if not os.path.exists(result_dir):
        os.system('mkdir -p ' + result_dir)
    args.result_dir = result_dir


def save_args(args, save_dir):
    args_path = os.path.join(save_dir, 'args.txt')
    with open(args_path, 'w') as fd:
        fd.write(str(args).replace(', ', ',\n'))


def prepare_train_args():
    args = get_train_args()
    get_train_model_dir(args)
    save_args(args, args.model_dir)
    return args


def prepare_test_args():
    args = get_test_args()
    get_test_result_dir(args)
    save_args(args, args.result_dir)
    return args


if __name__ == '__main__':
    args = get_test_args()
    get_test_result_dir(args)
    print(args.result_dir)
    exit()

    args = prepare_test_args()
    # train_args = get_train_args()
    test_args = get_test_args()

    parser = argparse.ArgumentParser()
    parser = parse_common_args(parser)

    args = parser.parse_args()
    print(args.gpus)
