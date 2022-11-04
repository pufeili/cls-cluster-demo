import torch
from torch.autograd import Variable
from data.data_entry import select_eval_loader
from model.model_entry import select_model
from options import prepare_test_args
from utils.logger import Recoder
import numpy as np
import cv2
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import pandas as pd

from utils.viz import label2rgb
from data.prepare_data import generate_dataloader


class Evaluator:
    def __init__(self):
        args = prepare_test_args()
        self.args = args
        self.model = select_model(args)
        self.model.load_state_dict(torch.load(args.load_model_path))
        self.model.eval()
        # self.val_loader = select_eval_loader(args)
        # process data and prepare dataloaders
        self.train_loader_source, self.train_loader_target, self.val_loader_target, self.val_loader_source = generate_dataloader(
            args)
        self.val_loader = self.val_loader_target

        self.recoder = Recoder()

    def eval(self):
        for i, data in enumerate(self.val_loader):
            img, pred, label = self.step(data)
            metrics = self.compute_metrics(pred, label)

            for key in metrics.keys():
                self.recoder.record(key, metrics[key])
            if i % self.args.viz_freq:
                self.viz_per_batch(img, pred, label, i)

        metrics = self.recoder.summary()
        result_txt_path = os.path.join(self.args.result_dir, 'result.txt')

        # write metrics to result dir,
        # you can also use pandas or other methods for better stats
        with open(result_txt_path, 'w') as fd:
            fd.write(str(metrics))

    def compute_metrics(self, pred, gt):
        # you can call functions in metrics.py
        l1 = (pred - gt).abs().mean()
        metrics = {
            'l1': l1
        }
        return metrics

    def viz_per_batch(self, img, pred, gt, step):
        # call functions in viz.py
        # here is an example about segmentation
        img_np = img[0].cpu().numpy().transpose((1, 2, 0))
        pred_np = label2rgb(pred[0].cpu().numpy())
        gt_np = label2rgb(gt[0].cpu().numpy())
        viz = np.concatenate([img_np, pred_np, gt_np], axis=1)
        viz_path = os.path.join(self.args.result_dir, "%04d.jpg" % step)
        cv2.imwrite(viz_path, viz)

    def step(self, data):
        img, label = data
        # warp input
        img = Variable(img).cuda()
        label = Variable(label).cuda()

        # compute output
        pred = self.model(img)
        return img, label, pred

    def draw_tsne(self, plot=1, save_fig=False):
        feats1 = []
        feats2 = []
        labels = []
        print('Obtaining features')
        for _, batch in tqdm(enumerate(self.val_loader)):
            img, label = batch
            # label = label.cuda()
            with torch.no_grad():
                feat1, feat2, pred = self.model(img.cuda())

            feats1.extend(torch.chunk(feat1, feat1.shape[0], dim=0))
            feats2.extend(torch.chunk(feat2, feat2.shape[0], dim=0))
            labels.extend(label.tolist())
        feats1 = torch.stack(feats1).squeeze()
        feats2 = torch.stack(feats2).squeeze()
        labels = torch.from_numpy(np.array(labels))
        if plot == 1:
            self.plot_embedding(feats1.cpu().numpy(), labels.numpy(), "feature 1 t-sne distribution", save_fig)
        elif plot == 2:
            self.plot_embedding(feats2.cpu().numpy(), labels.numpy(), "feature 2 t-sne distribution", save_fig)

    def plot_embedding(self, features, label, title, save_fig=False):
        print('Computing t-SNE embedding')
        tsne = TSNE(n_components=2, init='pca', random_state=0)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            x_tsne = tsne.fit_transform(features)
        print("Org data dimension is {}.\nEmbedded data dimension is {}.".format(features.shape[-1], x_tsne.shape[-1]))
        print("Number of samples: {}.".format(features.shape[0]))
        # Min-Max Normalization
        x_min, x_max = np.min(x_tsne, 0), np.max(x_tsne, 0)
        x_norm = (x_tsne - x_min) / (x_max - x_min)
        x_new = np.hstack((x_norm, label.reshape((-1, 1))))
        x_new = pd.DataFrame({'x': x_new[:, 0], 'y': x_new[:, 1], 'label': x_new[:, 2]})

        fig = plt.figure(figsize=(10, 10))
        ax1 = fig.add_subplot(111)
        for index in range(features.shape[0]):
            X = x_new.loc[x_new['label'] == index]['x']
            Y = x_new.loc[x_new['label'] == index]['y']
            plt.scatter(X, Y, cmap='brg', s=100)

        plt.xticks([])  # 去掉横坐标值
        plt.yticks([])  # 去掉横坐标值
        plt.title(title, fontsize=32, fontweight='normal', pad=20)
        if save_fig:
            # plt.subplots_adjust(top=0.95, bottom=0, right=1, left=0, hspace=0.02, wspace=0.02)  # 去白边
            plt.savefig(os.path.join(self.args.result_dir, str(title) + '.png'), format='png', dpi=300, bbox_inches='tight')
        plt.show()


def eval_main():
    evaluator = Evaluator()
    # evaluator.eval()
    evaluator.draw_tsne(plot=1, save_fig=False)


if __name__ == '__main__':
    eval_main()
