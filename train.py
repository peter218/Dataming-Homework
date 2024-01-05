import os
import argparse

import torch
import numpy as np
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader

from config import Config
from model import create_model
from loss_fn import get_loss_fn
from utils import save_result, load_pickle
from utils.visualization import plot_mean_std_curves
from utils.data import STS2DDataset
from scipy.spatial.distance import directed_hausdorff
from metric import hausdorff_distance_2d,intersection_over_union,dice_coefficient

def compute_dice(pred, mask):
    intersection = torch.sum(pred * mask)
    return 2.0 * intersection / (torch.sum(pred) + torch.sum(mask))


def compute_iou(pred, mask):
    intersection = torch.sum(pred * mask)
    union = torch.sum(pred) + torch.sum(mask) - intersection
    return intersection / union


def compute_hausdorff(pred, mask):
    # Ensure that pred and mask are numpy arrays on the CPU
    # 确保 pred 和 mask 是 torch.Tensor
    if not isinstance(pred, torch.Tensor):
        pred = torch.tensor(pred)
    if not isinstance(mask, torch.Tensor):
        mask = torch.tensor(mask)

    # 转换为二值图
    preds_binary = (pred > 0.5).int()
    masks_binary = (mask > 0.5).int()

    # 获取坐标点
    coords_pred = torch.nonzero(preds_binary == 1, as_tuple=False)
    coords_mask = torch.nonzero(masks_binary == 1, as_tuple=False)

    # 初始化最小距离为一个很大的数
    min_dist = float('inf')

    # 如果使用GPU，确保将数据移到GPU上
    if torch.cuda.is_available():
        coords_pred = coords_pred.cuda()
        coords_mask = coords_mask.cuda()

    # 分批计算点对之间的距离，逐个更新最小距离
    for point_pred in coords_pred:
        # 广播计算当前点与所有掩码点的曼哈顿距离
        dists = torch.sum(torch.abs(point_pred - coords_mask), dim=1)
        # 更新最小距离
        min_dist = min(min_dist, torch.min(dists).item())

    # 如果在GPU上计算，确保将结果移回CPU
    if torch.cuda.is_available():
        min_dist = torch.tensor(min_dist).cpu().item()

    return min_dist

class Engine:
    def __init__(self, cfg: Config):
        # 挂载一些重要参数
        self.epochs = cfg.EPOCHS
        self.start_epoch = cfg.START_EPOCH
        self.running_dir = cfg.RUNNING_DIR
        self.model_path = cfg.MODEL_PATH
        self.save_interval = cfg.SAVE_INTERVAL
        self.valid_interval = cfg.VALID_INTERVAL
        self.logger = cfg.logger

        # GPU硬件环境
        self.device = torch.device(cfg.DEVICE)

        # 模型配置
        self.model_name = f'{cfg.MODEL_ARCH}+{cfg.MODEL_ENCODER_NAME}'
        self.model = create_model(
            arch=cfg.MODEL_ARCH,
            encoder_name=cfg.MODEL_ENCODER_NAME,
            encoder_weights=cfg.MODEL_ENCODER_WEIGHTS,
            in_channels=3,
            classes=1,
            **cfg.MODEL_KWARGS,
        )
        if cfg.DEVICE_IDS is not None:
            self.model = torch.nn.DataParallel(self.model, device_ids=cfg.DEVICE_IDS)
        self.model.to(self.device)

        if cfg.MODEL_CHECKPOINT:
            self.load_model(cfg.MODEL_CHECKPOINT)

        # 训练配置
        self.criterion = get_loss_fn(cfg.LOSS_FN, **cfg.LOSS_FN_KWARGS).to(self.device)
        self.optimizer = cfg.OPTIMIZER(self.model.parameters(), lr=cfg.LEARNING_RATE, **cfg.OPTIM_KWARGS)
        if cfg.SCHEDULER is not None:
            self.scheduler = cfg.SCHEDULER(self.optimizer, **cfg.SCHEDULER_KWARGS)
        else:
            self.scheduler = None

        self.best_score = 1e6

        # 评估配置
        self.threshold = cfg.THRESHOLD
        self.metric = cfg.EVALUATION_METRIC
        self.metric_key = cfg.METRIC_KEYS


    def train(self, train_dataloader, validation=False, valid_dataloader=None):
        self.logger.info('START TRAINING')

        losses = {'train': [], 'valid': []}
        for ep in range(self.start_epoch, self.epochs):
            # 训练
            train_result = self.train_epoch(train_dataloader)
            losses['train'].append(train_result['loss'])

            self.logger.info('[train] epoch:{} loss: {:.6} lr: {:.8}'.format(
                ep + 1, train_result['loss'], self.optimizer.param_groups[0]['lr'],
            ))

            if self.scheduler is not None:
                self.scheduler.step()

            if (ep + 1) % self.save_interval == 0:
                self.save_model(f'{self.model_name}+epoch{ep + 1:03d}.pth')

            # 验证
            if not validation:
                if train_result['loss'] < self.best_score:
                    self.best_score = train_result['loss']
                    self.save_model(f'{self.model_name}+best.pth')
            elif (ep + 1) % self.valid_interval == 0:
                # todo: a better validation method
                valid_result = self.valid_epoch(valid_dataloader)
                losses['valid'].append(valid_result['loss'])

                self.logger.info('[valid] epoch:{} loss: {:.6} dice:{:.6} iou: {:.6}  hausdorff:{:.6}  scores:{:.6}'.format(
                    ep + 1, valid_result['loss'],valid_result['dice_score'],valid_result['iou_score'],valid_result['hausdorff_score'],valid_result['score']
                ))

                if valid_result['loss'] < self.best_score:
                    self.best_score = valid_result['loss']
                    self.save_model(f'{self.model_name}+best.pth')

        save_result(losses, os.path.join(self.running_dir, 'loss.pkl'))

    def train_epoch(self, dataloader):
        self.model.train()

        ep_loss = []
        for data in dataloader:
            images = data['image'].to(self.device)
            masks = data['mask'].to(self.device)

            preds = self.model(images)
            loss = self.criterion(preds, masks)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            ep_loss.append(loss.detach().cpu().numpy())

        return {'loss': np.mean(ep_loss)}

    @torch.no_grad()
    def valid_epoch(self, dataloader):
        self.model.eval()
        # todo: a better validation method
        scores = []
        dice_scores = []
        iou_scores = []
        ep_loss = []
        hausdorff_scores = []
        for data in dataloader:
            images = data['image'].to(self.device)
            masks = data['mask'].to(self.device)

            preds = self.model(images)
            preds = preds.squeeze(1)
            loss = self.criterion(preds, masks)

            ep_loss.append(loss.cpu().numpy())
            preds_binary = preds > 0.5  # Threshold to convert to binary mask
            dice_score = dice_coefficient(preds_binary, masks)
            iou_score = intersection_over_union(preds_binary, masks)
            hausdorff_distance = hausdorff_distance_2d(preds_binary, masks)
            torch.tensor(hausdorff_distance).to(self.device)
            score = 0.4 * dice_score + 0.3 * iou_score + 0.3 * (1 - hausdorff_distance)
            dice_scores.append(dice_score.cpu().numpy())
            iou_scores.append(iou_score.cpu().numpy())
            hausdorff_scores.append(hausdorff_distance.cpu().numpy())
            scores.append(score.cpu().numpy())
        return {'loss': np.mean(ep_loss),'dice_score':np.mean(dice_scores),'iou_score':np.mean(iou_scores),'hausdorff_score':np.mean(hausdorff_scores),'score': np.mean(scores)}

    def save_model(self, filename: str):
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        torch.save(self.model.state_dict(),
                   os.path.join(self.model_path, filename))
        self.logger.info(f'Successfully save model to {os.path.join(self.model_path, filename)}')

    def load_model(self, file_path: str):
        self.model.load_state_dict(torch.load(file_path))
        self.logger.info(f'Successfully load model {file_path}')

    @torch.no_grad()
    def evaluate(self, dataloader, model: str = 'best'):
        self.logger.info('START EVALUATING')

        if isinstance(model, torch.nn.Module):
            self.model = model.to(self.device)
        elif model == 'best':
            self.load_model(os.path.join(self.model_path, f'{self.model_name}+best.pth'))
        else:
            self.load_model(model)
        self.model.eval()

        results = {
            key: 0 for key in self.metric_key
        }
        num_sample = 0
        for data in dataloader:
            images = data['image'].to(self.device)
            masks = data['mask'].to(self.device)
            num_sample += len(images)

            preds = self.model(images)
            preds = (torch.sigmoid(preds) > self.threshold).float().squeeze(dim=1)

            for key in self.metric_key:
                output = self.metric[key](preds, masks)
                results[key] += output.sum().cpu().numpy()

        for key in self.metric_key:
            results[key] /= num_sample
            self.logger.info(f'{key:>10}: {results[key]:.6f}')

        return








    # todo: a better evaluation method
    def eval_epoch(self):
        pass


def k_fold_cross_validate(cfg: Config):
    import pandas as pd

    def k_fold_data_split():
        data_train_dir = os.path.join(cfg.DATA_DIR, 'image')
        filenames = np.array([filename for filename in os.listdir(data_train_dir)
                              if filename[-3:] == 'png'])
        kf = KFold(n_splits=cfg.K_FOLD)
        for train_idx, valid_idx in kf.split(filenames):
            train_data = STS2DDataset(cfg.DATA_DIR, filenames[train_idx], transform=cfg.TRAIN_TRANSFORMS)
            valid_data = STS2DDataset(cfg.DATA_DIR, filenames[valid_idx], transform=cfg.VALID_TRANSFORMS)
            train_dataloader = DataLoader(train_data, batch_size=cfg.BATCH_SIZE, shuffle=True,
                                          num_workers=cfg.NUM_WORKER)
            valid_dataloader = DataLoader(valid_data, batch_size=cfg.BATCH_SIZE, shuffle=False,
                                          num_workers=cfg.NUM_WORKER)
            yield train_dataloader, valid_dataloader

    models = ['best'] + [f'EP-{ep}' for ep in range(cfg.SAVE_INTERVAL, cfg.EPOCHS + 1, cfg.SAVE_INTERVAL)]
    evaluation = {
        f'{model}-{key}': []
        for model in models
        for key in cfg.METRIC_KEYS
    }

    for k, (train_loader, valid_loader) in enumerate(k_fold_data_split()):
        cfg.change_running_dir(os.path.join('run1', cfg.LAB_ID, f'fold-{k + 1}'))
        cfg.logger.info(f'\n\n--- Fold {k + 1}/{cfg.K_FOLD} ---\n')
        engine = Engine(cfg)
        # engine.train(train_loader, True, valid_loader)
        engine.train(train_loader)

        result = engine.evaluate(valid_loader, model='best')
        for key in cfg.METRIC_KEYS:
            evaluation[f'best-{key}'].append(result[key])

        for ep in range(cfg.SAVE_INTERVAL, cfg.EPOCHS + 1, cfg.SAVE_INTERVAL):
            result = engine.evaluate(
                valid_loader, model=os.path.join(cfg.MODEL_PATH, f'{engine.model_name}+epoch{ep:03d}.pth'))
            for key in cfg.METRIC_KEYS:
                evaluation[f'EP-{ep}-{key}'].append(result[key])

    cfg.change_running_dir(os.path.join('run', cfg.LAB_ID))

    df = pd.DataFrame.from_dict(evaluation)
    df.index = pd.Series([f'fold-{k+1}' for k in range(cfg.K_FOLD)])
    col_mean = df.mean()
    col_std = df.std()
    df.loc['mean'] = col_mean
    df.loc['std'] = col_std
    df.to_csv(os.path.join(cfg.RUNNING_DIR, 'evaluation.csv'))

    losses = {'train': [], 'valid': []}
    for k in range(cfg.K_FOLD):
        loss = load_pickle(os.path.join(cfg.RUNNING_DIR, f'fold-{k + 1}', 'loss.pkl'))
        losses['train'].append(loss['train'])
        losses['valid'].append(loss['valid'])
    plot_mean_std_curves(
        losses,
        strids={'train': 1, 'valid': cfg.VALID_INTERVAL},
        labels={'train': 'train loss', 'valid': 'valid loss'},
        filename=os.path.join(cfg.RUNNING_DIR, 'loss_curve.png'),
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--k_fold', type=bool, default=False)
    args = parser.parse_args()

    config = Config()
    if args.k_fold:
        # 使用k折交叉验证进行实验
        config.logger.info('--- K-FOLD CROSS VALIDATION ---')
        k_fold_cross_validate(config)
    else:
        # 在全部数据集上训练模型
        config.logger.info('--- TRAIN ON FULL DATA ---')
        image_dir = os.path.join(config.DATA_DIR, 'image')
        files = [file for file in os.listdir(image_dir) if file[-3:] == 'png']
        t_dataset = STS2DDataset(config.DATA_DIR, files, transform=config.TRAIN_TRANSFORMS)
        v_dataset = STS2DDataset(config.DATA_DIR, files, transform=config.VALID_TRANSFORMS)
        t_loader = DataLoader(t_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKER)
        v_loader = DataLoader(v_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKER)

        runner = Engine(config)
        runner.train(t_loader,validation=True,valid_dataloader=v_loader)

