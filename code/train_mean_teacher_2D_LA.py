import argparse
import logging
import os
import random
import shutil
import sys
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn import BCEWithLogitsLoss
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm import tqdm

from dataloaders import utils
from dataloaders.dataset import (BaseDataSets_LA, RandomGenerator,
                                 TwoStreamBatchSampler)
from networks.net_factory import net_factory
from utils import losses, metrics, ramps
from val_2D import test_single_volume_LA
from torchvision.transforms.functional import adjust_brightness, adjust_contrast
from chamferdist import ChamferDistance
import scipy.ndimage as ndimage

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='D:\Code\CPL\data\LA', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='LA/Mean_Teacher', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='unet', help='model_name')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=4,
                    help='batch_size per gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list,  default=[112, 112],
                    help='patch size of network input')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--num_classes', type=int,  default=2,
                    help='output channel of network')

# label and unlabel
parser.add_argument('--labeled_bs', type=int, default=2,
                    help='labeled_batch_size per gpu')
parser.add_argument('--labeled_num', type=int, default=8,
                    help='labeled data')
# costs
parser.add_argument('--ema_decay', type=float,  default=0.99, help='ema_decay')
parser.add_argument('--consistency_type', type=str,
                    default="mse", help='consistency_type')
parser.add_argument('--consistency', type=float,
                    default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float,
                    default=200.0, help='consistency_rampup')
args = parser.parse_args()


def patients_to_slices(dataset, patiens_num):
    ref_dict = None
    if "LA" in dataset:
        ref_dict = {"1": 32, "4": 352, "8": 704,
                    "14": 256, "21": 396, "28": 512, "35": 664, "70": 1312}
    elif "Prostate":
        ref_dict = {"2": 27, "4": 53, "8": 120,
                    "12": 179, "16": 256, "21": 312, "42": 623}
    else:
        print("Error")
    return ref_dict[str(patiens_num)]

def cosine_similarity(x, y):
    x_mean = x.view(x.size(0), -1).mean(dim=1)  # (batch_size, height * width) -> (batch_size)
    y_mean = y.view(y.size(0), -1).mean(dim=1)  # (batch_size, height * width) -> (batch_size)
    cos_sim = F.cosine_similarity(x_mean, y_mean, dim=0)  # (batch_size, )
    return cos_sim

def pixel_cosine_loss(student_features, teacher_features, threshold=0.5):
    total_loss = 0
    batch_size, num_clusters, height, width = student_features.shape

    for student_cluster_idx in range(num_clusters):
        student_cluster = student_features[:, student_cluster_idx, :, :]
        cluster_loss = 0
        for teacher_cluster_idx in range(num_clusters):
            teacher_cluster = teacher_features[:, teacher_cluster_idx, :, :]

            cos_sim = cosine_similarity(student_cluster, teacher_cluster)

            positive_mask = cos_sim > threshold  # 相似度大于阈值的为正样本
            negative_mask = cos_sim <= threshold  # 相似度小于等于阈值的为负样本
            positive_count = positive_mask.sum().item()
            negative_count = negative_mask.sum().item()

            if positive_count > 0:
                positive_loss = (1 - cos_sim * positive_mask.float()).mean()
            else:
                positive_loss = 0
            if negative_count > 0:
                negative_loss = ((1 - cos_sim) * negative_mask.float()).mean()
            else:
                negative_loss = 0
            cluster_loss += (positive_loss + negative_loss)
        total_loss += cluster_loss / num_clusters ** 2

    return total_loss

def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


def train(args, snapshot_path):
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size
    max_iterations = args.max_iterations

    def create_model(ema=False):
        # Network definition
        model = net_factory(net_type=args.model, in_chns=1,
                            class_num=num_classes)
        if ema:
            for param in model.parameters():
                param.detach_()
        return model

    model = create_model()
    ema_model = create_model(ema=True)

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    db_train = BaseDataSets_LA(base_dir=args.root_path, split="train", num=None, transform=transforms.Compose([
        RandomGenerator(args.patch_size)
    ]))
    db_val = BaseDataSets_LA(base_dir=args.root_path, split="test")

    total_slices = len(db_train)
    labeled_slice = patients_to_slices(args.root_path, args.labeled_num)
    print("Total silices is: {}, labeled slices is: {}".format(
        total_slices, labeled_slice))
    labeled_idxs = list(range(0, labeled_slice))
    unlabeled_idxs = list(range(labeled_slice, total_slices))
    batch_sampler = TwoStreamBatchSampler(
        labeled_idxs, unlabeled_idxs, batch_size, batch_size-args.labeled_bs)

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler,
                             num_workers=0, pin_memory=True, worker_init_fn=worker_init_fn)

    model.train()

    valloader = DataLoader(db_val, batch_size=1, shuffle=False,
                           num_workers=1)

    optimizer = optim.SGD(model.parameters(), lr=base_lr,
                          momentum=0.9, weight_decay=0.0001)
    ce_loss = CrossEntropyLoss()
    dice_loss = losses.DiceLoss(num_classes)

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)

    def soft_brightness_augmentation(image):
        brightness_factor = random.uniform(0.5, 1.5)
        return adjust_brightness(image, brightness_factor)

    def soft_contrast_augmentation(image):
        contrast_factor = random.uniform(0.5, 1.5)
        return adjust_contrast(image, contrast_factor)

    def random_soft_augmentation(image):
        augmentations = [
            soft_brightness_augmentation,
            soft_contrast_augmentation,
            lambda x: add_gaussian_noise(x)
        ]
        augmentation = random.choice(augmentations)
        return augmentation(image)

    def add_gaussian_noise(image, sigma=0.1):
        noise = torch.clamp(torch.randn_like(image) * sigma, -0.2, 0.2)
        return image + noise

    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):

            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            unlabeled_volume_batch = volume_batch[args.labeled_bs:]

            ema_inputs = random_soft_augmentation(unlabeled_volume_batch)

            outputs, pred_mask1, out1, semseg1 = model(volume_batch)
            outputs_soft_mask = torch.softmax(semseg1, dim=1)
            outputs_soft = torch.softmax(outputs, dim=1)
            with torch.no_grad():
                ema_output, pred_mask2, out2, semseg2 = ema_model(ema_inputs)
                ema_output_soft = torch.softmax(ema_output, dim=1)
                ema_output_soft_mask = torch.softmax(semseg2, dim=1)

            loss_ce = ce_loss(outputs[:args.labeled_bs],
                              label_batch[:][:args.labeled_bs].long())
            loss_dice = dice_loss(
                outputs_soft[:args.labeled_bs], label_batch[:args.labeled_bs].unsqueeze(1))
            supervised_loss = 0.5 * (loss_dice + loss_ce)
            loss_ce_mask = ce_loss(semseg1[:args.labeled_bs],
                              label_batch[:][:args.labeled_bs].long())
            loss_dice_mask = dice_loss(
                outputs_soft_mask[:args.labeled_bs], label_batch[:args.labeled_bs].unsqueeze(1))
            seg_loss = 0.5 * (loss_ce_mask + loss_dice_mask)
            consistency_weight = get_current_consistency_weight(iter_num//150)
            if iter_num < 1000:
                consistency_loss = 0.0
                loss_contr = 0.0
                cons_mask = 0.0
            else:
                loss_contr = pixel_cosine_loss(pred_mask2, pred_mask1[args.labeled_bs:])
                consistency_loss = torch.mean(
                    (outputs_soft[args.labeled_bs:]-ema_output_soft)**2)
                cons_mask = torch.mean(
                    (outputs_soft[args.labeled_bs:]-ema_output_soft_mask)**2)

            loss = supervised_loss + seg_loss + consistency_weight * (consistency_loss + cons_mask + loss_contr)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            update_ema_variables(model, ema_model, args.ema_decay, iter_num)
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)
            writer.add_scalar('info/loss_dice', loss_dice, iter_num)
            writer.add_scalar('info/loss_ce_mask', loss_ce_mask, iter_num)
            writer.add_scalar('info/loss_dice_mask', loss_dice_mask, iter_num)
            writer.add_scalar('info/consistency_loss',
                              consistency_loss, iter_num)
            writer.add_scalar('info/loss_contr', loss_contr, iter_num)
            writer.add_scalar('info/cons_mask', cons_mask, iter_num)
            writer.add_scalar('info/consistency_weight',
                              consistency_weight, iter_num)

            logging.info(
                'iteration %d : loss : %f, loss_ce: %f, loss_dice: %f, loss_boundary: %f, loss_ce_mask: %f, loss_dice_mask: %f, consistency_loss: %f, loss_contr: %f, cons_mask: %f' %
                (iter_num, loss.item(), loss_ce.item(), loss_dice.item(), loss_ce_mask, loss_dice_mask, consistency_loss, loss_contr, cons_mask))

            if iter_num % 20 == 0:
                image = volume_batch[1, 0:1, :, :]
                writer.add_image('train/Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(
                    outputs, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction',
                                 outputs[1, ...] * 50, iter_num)
                labs = label_batch[1, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)

            if iter_num > 0 and iter_num % 200 == 0:
                model.eval()
                metric_list = 0.0
                for i_batch, sampled_batch in enumerate(valloader):
                    metric_i = test_single_volume_LA(
                        sampled_batch["image"], sampled_batch["label"], model, classes=num_classes)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)
                for class_i in range(num_classes-1):
                    writer.add_scalar('info/val_{}_dice'.format(class_i+1),
                                      metric_list[class_i, 0], iter_num)
                    writer.add_scalar('info/val_{}_hd95'.format(class_i+1),
                                      metric_list[class_i, 1], iter_num)

                performance = np.mean(metric_list, axis=0)[0]

                mean_hd95 = np.mean(metric_list, axis=0)[1]
                writer.add_scalar('info/val_mean_dice', performance, iter_num)
                writer.add_scalar('info/val_mean_hd95', mean_hd95, iter_num)

                if performance > best_performance:
                    best_performance = performance
                    save_mode_path = os.path.join(snapshot_path,
                                                  'iter_{}_dice_{}.pth'.format(
                                                      iter_num, round(best_performance, 4)))
                    save_best = os.path.join(snapshot_path,
                                             '{}_best_model.pth'.format(args.model))
                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best)

                logging.info(
                    'iteration %d : mean_dice : %f mean_hd95 : %f' % (iter_num, performance, mean_hd95))
                model.train()

            if iter_num % 3000 == 0:
                save_mode_path = os.path.join(
                    snapshot_path, 'iter_' + str(iter_num) + '.pth')
                torch.save(model.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))

            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()
    return "Training Finished!"


if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    snapshot_path = "D:\Code\CPL\model\{}_{}_labeled" \
                    "\{}".format(
        args.exp, args.labeled_num, args.model)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copytree('.', snapshot_path + '/code',
                    shutil.ignore_patterns(['.git', '__pycache__']))

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    train(args, snapshot_path)
