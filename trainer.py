import argparse
import logging
import os
import random
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import DiceLoss
from torchvision import transforms
import torch.nn.functional as F

def build_dataset(args, split):
    from datasets.dataset_synapse import Synapse_dataset, RandomGenerator
    from datasets.dataset_busi import BUSI_dataset, RandomGeneratorBUSI

    if args.dataset == 'Synapse':
        ds = Synapse_dataset(
            base_dir=args.root_path,
            list_dir=args.list_dir,
            split=split,
            transform=transforms.Compose([RandomGenerator(output_size=[args.img_size, args.img_size])])
        )
    elif args.dataset == 'BUSI':
        ds = BUSI_dataset(
            base_dir=args.root_path,
            list_dir=args.list_dir,
            split=split,
            transform=transforms.Compose([RandomGeneratorBUSI(output_size=[args.img_size, args.img_size])])
        )
    else:
        raise ValueError(f'Unknown dataset: {args.dataset}')
    return ds


@torch.no_grad()
def eval_val_dice(model, loader, num_classes, device="cuda"):
    model.eval()
    dices = []
    for batch in loader:
        imgs = batch['image'].to(device)
        gts  = batch['label'].to(device).long()  # [B,H,W]
        logits = model(imgs)                     # [B,C,H,W]
        preds = torch.argmax(F.softmax(logits, dim=1), dim=1)  # [B,H,W]

        if num_classes == 2:
            # foreground (class=1) Dice
            p = (preds == 1).float()
            t = (gts  == 1).float()
            inter = (p * t).sum(dim=(1,2))
            denom = p.sum(dim=(1,2)) + t.sum(dim=(1,2)) + 1e-6
            dice  = (2.0 * inter / denom).mean()
        else:
            # mean Dice over classes 1..C-1 (exclude background)
            dice_per_img = []
            for c in range(1, num_classes):
                p = (preds == c).float()
                t = (gts  == c).float()
                inter = (p * t).sum(dim=(1,2))
                denom = p.sum(dim=(1,2)) + t.sum(dim=(1,2)) + 1e-6
                dice_c = 2.0 * inter / denom
                dice_per_img.append(dice_c)
            if len(dice_per_img) == 0:
                continue
            dice = torch.stack(dice_per_img, dim=0).mean()  # mean over classes and batch
        dices.append(dice)
    model.train()
    if len(dices) == 0:
        return 0.0
    return torch.stack(dices).mean().item()

# def trainer_synapse(args, model, snapshot_path):
#     from datasets.dataset_synapse import Synapse_dataset, RandomGenerator
#     logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
#                         format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
#     logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
#     logging.info(str(args))
#     base_lr = args.base_lr
#     num_classes = args.num_classes
#     if args.dataset == 'BUSI':
#         assert num_classes == 2, "For BUSI set --num_classes 2"

#     batch_size = args.batch_size * args.n_gpu
#     # max_iterations = args.max_iterations
#     # db_train = Synapse_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="train",
#     #                            transform=transforms.Compose(
#     #                                [RandomGenerator(output_size=[args.img_size, args.img_size])]))
#     db_train = build_dataset(args, split="train")
#     print("The length of train set is: {}".format(len(db_train)))

#     def worker_init_fn(worker_id):
#         random.seed(args.seed + worker_id)

#     trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True,
#                              worker_init_fn=worker_init_fn)
#     if args.n_gpu > 1:
#         model = nn.DataParallel(model)
#     model.train()
#     ce_loss = CrossEntropyLoss()
#     dice_loss = DiceLoss(num_classes)
#     optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
#     writer = SummaryWriter(snapshot_path + '/log')
#     iter_num = 0
#     max_epoch = args.max_epochs
#     max_iterations = args.max_epochs * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
#     logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
#     best_performance = 0.0
#     iterator = tqdm(range(max_epoch), ncols=70)
#     for epoch_num in iterator:
#         for i_batch, sampled_batch in enumerate(trainloader):
#             image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
#             image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
#             outputs = model(image_batch)
#             loss_ce = ce_loss(outputs, label_batch[:].long())
#             loss_dice = dice_loss(outputs, label_batch, softmax=True)
#             loss = 0.5 * loss_ce + 0.5 * loss_dice
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#             lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
#             for param_group in optimizer.param_groups:
#                 param_group['lr'] = lr_

#             iter_num = iter_num + 1
#             writer.add_scalar('info/lr', lr_, iter_num)
#             writer.add_scalar('info/total_loss', loss, iter_num)
#             writer.add_scalar('info/loss_ce', loss_ce, iter_num)

#             logging.info('iteration %d : loss : %f, loss_ce: %f' % (iter_num, loss.item(), loss_ce.item()))

#             # if iter_num % 20 == 0:
#             #     image = image_batch[1, 0:1, :, :]
#             #     image = (image - image.min()) / (image.max() - image.min())
#             #     writer.add_image('train/Image', image, iter_num)
#             #     outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
#             #     writer.add_image('train/Prediction', outputs[1, ...] * 50, iter_num)
#             #     labs = label_batch[1, ...].unsqueeze(0) * 50
#             #     writer.add_image('train/GroundTruth', labs, iter_num)

#             if iter_num % 20 == 0:
#                 # image logging (handle C==1 or C==3)
#                 img = image_batch[1]  # [C,H,W]
#                 # normalize per-sample for visualization
#                 imin, imax = img.min(), img.max()
#                 if (imax - imin) > 1e-6:
#                     img = (img - imin) / (imax - imin)

#                 if img.shape[0] == 1:
#                     writer.add_image('train/Image', img, iter_num)                # [1,H,W]
#                 else:
#                     writer.add_image('train/Image', img[:3, ...], iter_num)       # [3,H,W]

#                 # prediction & GT
#                 preds = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)  # [B,1,H,W]
#                 writer.add_image('train/Prediction', preds[1] * 50, iter_num)
#                 labs = label_batch[1].unsqueeze(0) * 50  # [1,H,W]
#                 writer.add_image('train/GroundTruth', labs, iter_num)


#         save_interval = 50  # int(max_epoch/6)
#         if epoch_num > int(max_epoch / 2) and (epoch_num + 1) % save_interval == 0:
#             save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
#             torch.save(model.state_dict(), save_mode_path)
#             logging.info("save model to {}".format(save_mode_path))

#         if epoch_num >= max_epoch - 1:
#             save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
#             torch.save(model.state_dict(), save_mode_path)
#             logging.info("save model to {}".format(save_mode_path))
#             iterator.close()
#             break

#     writer.close()
#     return "Training Finished!"

def trainer_synapse(args, model, snapshot_path):
    # logging
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    # ensure repo root on sys.path (useful in Colab)
    ROOT = os.path.dirname(os.path.abspath(__file__))
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)

    base_lr = args.base_lr
    num_classes = args.num_classes
    if args.dataset == 'BUSI':
        assert num_classes == 2, "For BUSI set --num_classes 2"

    batch_size = args.batch_size * args.n_gpu

    # --- datasets & loaders ---
    db_train = build_dataset(args, split="train")
    db_val   = build_dataset(args, split="test")
    print("The length of train set is: {}".format(len(db_train)))
    print("The length of val   set is: {}".format(len(db_val)))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=8,
                             pin_memory=True, worker_init_fn=worker_init_fn)
    valloader   = DataLoader(db_val,   batch_size=batch_size, shuffle=False, num_workers=4,
                             pin_memory=True, worker_init_fn=worker_init_fn)

    # --- model & opt ---
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()
    ce_loss   = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))

    # --- early stopping state ---
    PATIENCE = 100
    best_val_dice = -1.0
    best_epoch = -1
    epochs_no_improve = 0
    best_ckpt_path = os.path.join(snapshot_path, 'best_model.pth')

    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        # -------- train one epoch --------
        epoch_loss = 0.0
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
            outputs = model(image_batch)
            loss_ce = ce_loss(outputs, label_batch.long())
            loss_dice = dice_loss(outputs, label_batch, softmax=True)
            loss = 0.5 * loss_ce + 0.5 * loss_dice

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num += 1
            epoch_loss += loss.item()

            writer.add_scalar('train/lr', lr_, iter_num)
            writer.add_scalar('train/total_loss', loss.item(), iter_num)
            writer.add_scalar('train/loss_ce', loss_ce.item(), iter_num)
            writer.add_scalar('train/loss_dice', loss_dice.item(), iter_num)

            if iter_num % 20 == 0:
                logging.info('iter %d : loss %.6f | ce %.6f | dice %.6f' %
                         (iter_num, loss.item(), loss_ce.item(), loss_dice.item()))

            if iter_num % 20 == 0:
                # image logging (handle C==1 or C==3)
                img = image_batch[1]
                imin, imax = img.min(), img.max()
                if (imax - imin) > 1e-6:
                    img = (img - imin) / (imax - imin)
                if img.shape[0] == 1:
                    writer.add_image('train/Image', img, iter_num)
                else:
                    writer.add_image('train/Image', img[:3, ...], iter_num)
                preds = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction', preds[1] * 50, iter_num)
                labs = label_batch[1].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)

        # -------- validate --------
        val_dice = eval_val_dice(model, valloader, num_classes, device="cuda")
        writer.add_scalar('val/dice', val_dice, epoch_num)
        logging.info(f"Epoch {epoch_num} | train_loss {epoch_loss/len(trainloader):.6f} | val_dice {val_dice:.4f}")

        # --- save best & early stop ---
        improved = val_dice > best_val_dice + 1e-6
        if improved:
            best_val_dice = val_dice
            best_epoch = epoch_num
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_ckpt_path)
            logging.info(f" New best Dice {best_val_dice:.4f} at epoch {epoch_num}. Saved {best_ckpt_path}")
        else:
            epochs_no_improve += 1
            logging.info(f"No improvement for {epochs_no_improve} epoch(s). Best {best_val_dice:.4f} @ epoch {best_epoch}")

        # periodic save (optional)
        save_interval = 50
        if epoch_num > int(max_epoch / 2) and (epoch_num + 1) % save_interval == 0:
            save_mode_path = os.path.join(snapshot_path, f'epoch_{epoch_num}.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info(f"Checkpoint saved to {save_mode_path}")

        # trigger early stop
        if epochs_no_improve >= PATIENCE:
            logging.info(f"Early stopping triggered after {PATIENCE} epochs without improvement.")
            break

        # last epoch save (in case we never improved)
        if epoch_num >= max_epoch - 1:
            last_path = os.path.join(snapshot_path, f'epoch_{epoch_num}.pth')
            torch.save(model.state_dict(), last_path)
            logging.info(f"Training finished. Last checkpoint saved to {last_path}")
            iterator.close()
            break

    writer.close()
    logging.info(f"Best val Dice: {best_val_dice:.4f} at epoch {best_epoch}")
    return "Training Finished!"
