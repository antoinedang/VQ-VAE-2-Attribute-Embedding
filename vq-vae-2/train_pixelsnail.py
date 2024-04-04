import argparse

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from model_definitions import *

try:
    from apex import amp

except ImportError:
    amp = None

from dataset import LMDBDataset
from scheduler import CycleScheduler

from tiff_dataset import GENRES


def train(args, epoch, loader, model, optimizer, scheduler, device, hier):
    loader = tqdm(loader)

    criterion = nn.CrossEntropyLoss()

    for i, (label, top, bottom, filename) in enumerate(loader):
        
        model.zero_grad()

        top = top.to(device)

        if hier == 'top':
            target = top
            out, _ = model(top)

        elif hier == 'bottom':
            bottom = bottom.to(device)
            target = bottom
            out, _ = model(bottom, condition=top)

        loss = criterion(out, target)
        loss.backward()

        if scheduler is not None:
            scheduler.step()
        optimizer.step()

        _, pred = out.max(1)
        correct = (pred == target).float()
        accuracy = correct.sum() / target.numel()

        lr = optimizer.param_groups[0]['lr']

        loader.set_description(
            (
                f'epoch: {epoch + 1}; loss: {loss.item():.5f}; '
                f'acc: {accuracy:.5f}; lr: {lr:.5f}'
            )
        )


class PixelTransform:
    def __init__(self):
        pass

    def __call__(self, input):
        ar = np.array(input)

        return torch.from_numpy(ar).long()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, default=2)
    parser.add_argument('--epoch', type=int, default=500)
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--checkpoint-folder', type=str, default="checkpoints")
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--amp', type=str, default='O0')
    parser.add_argument('--sched', type=str)
    parser.add_argument('path', type=str)

    args = parser.parse_args()

    print(args)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu": print("WARN: CUDA not available. Training will take very long.")
    

    for class_i in range(args.num_classes):
        for hier in ['top', 'bottom']:

            dataset = LMDBDataset(args.path, desired_class_label=class_i)
            
            loader = DataLoader(
                dataset, batch_size=args.batch, shuffle=True, num_workers=0, drop_last=True
            )

            if hier == 'top':
                model = getPixelSnailTop()
            elif hier == 'bottom':
                model = getPixelSnailBottom()

            model = model.to(device)
            optimizer = optim.Adam(model.parameters(), lr=args.lr)

            if amp is not None:
                model, optimizer = amp.initialize(model, optimizer, opt_level=args.amp)

            model = nn.DataParallel(model)
            model = model.to(device)

            scheduler = None
            if args.sched == 'cycle':
                scheduler = CycleScheduler(
                    optimizer, args.lr, n_iter=len(loader) * args.epoch, momentum=None
                )

            for i in range(args.epoch):
                train(args, i, loader, model, optimizer, scheduler, device, hier)
                torch.save(
                    {'model': model.module.state_dict(), 'args': args},
                    f'{args.checkpoint_folder}/pixelsnail_{GENRES[class_i]}_{hier}_{str(i + 1).zfill(3)}.pt',
                )
