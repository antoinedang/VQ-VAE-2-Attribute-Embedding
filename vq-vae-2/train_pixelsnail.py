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


criterion = nn.CrossEntropyLoss()

def train(args, epoch, loader, model, optimizer, scheduler, device, hier):
    loader = tqdm(loader)

    iters = 0
    loss_sum = 0
    acc_sum = 0

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
        
        top.detach().cpu()
        bottom.detach().cpu()
        del top
        del bottom

        acc_sum += accuracy
        loss_sum += loss.item()

        loader.set_description(
            (
                f'epoch: {epoch + 1}; loss: {loss.item():.5f}; '
                f'acc: {accuracy:.5f}; lr: {lr:.5f}'
            )
        )
        
        iters += 1
        
    return loss_sum / iters, acc_sum / iters


class PixelTransform:
    def __init__(self):
        pass

    def __call__(self, input):
        ar = np.array(input)

        return torch.from_numpy(ar).long()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--checkpoint-folder', type=str, default="checkpoints")
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--amp', type=str, default='O0')
    parser.add_argument('--sched', type=str)
    parser.add_argument('path', type=str)

    args = parser.parse_args()

    print(args)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu": print("WARN: CUDA not available. Training will take very long.")
    
    model = None

    for class_i in [1]:#range(args.num_classes):
        for hier in ['bottom', 'top']:

            dataset = LMDBDataset(args.path, desired_class_label=class_i)
            train_size = int(0.9 * len(dataset))
            test_size = len(dataset) - train_size
            train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
            
            loader = DataLoader(
                train_dataset, batch_size=args.batch, shuffle=True, num_workers=0, drop_last=True
            )

            del model
            torch.cuda.empty_cache()

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
                
            min_test_loss = 1000
            max_test_acc = 0

            for i in range(args.epoch):
                if i == 0:
                    with open(f'{args.checkpoint_folder}/pixelsnail_{GENRES[class_i]}_{hier}_metrics.csv', 'w+') as file:
                        file.write(f'epoch,avg_test_loss,avg_test_acc,avg_train_loss,avg_train_acc\n')
                avg_train_loss, avg_train_acc = train(args, i, loader, model, optimizer, scheduler, device, hier)
                
                model.eval()
                iters = 0
                avg_test_loss = 0
                avg_test_acc = 0

                for _, top, bottom, _ in test_dataset:
                    
                    model.zero_grad()

                    top = top.to(device).unsqueeze(0)
                    if hier == 'top':
                        target = top
                        out, _ = model(top)
                    elif hier == 'bottom':
                        bottom = bottom.to(device).unsqueeze(0)
                        target = bottom
                        out, _ = model(bottom, condition=top)

                    loss = criterion(out, target)

                    _, pred = out.max(1)
                    correct = (pred == target).float()
                    accuracy = correct.sum() / target.numel()

                    top.detach().cpu()
                    bottom.detach().cpu()
                    del top
                    del bottom

                    avg_test_acc += accuracy
                    avg_test_loss += loss.item()

                    iters += 1
                
                avg_test_acc = avg_test_acc / iters
                avg_test_loss = avg_test_loss / iters
                
                with open(f'{args.checkpoint_folder}/pixelsnail_{GENRES[class_i]}_{hier}_metrics.csv', 'a+') as file:
                    file.write(f'{i+1},{avg_test_loss},{avg_test_acc},{avg_train_loss},{avg_train_acc}\n')
                
                if avg_test_acc > max_test_acc or avg_test_loss < min_test_loss:
                    torch.save(
                        {'model': model.module.state_dict(), 'args': args},
                        f'{args.checkpoint_folder}/pixelsnail_{GENRES[class_i]}_{hier}_{str(i + 1).zfill(3)}.pt',
                    )
                    
                min_test_loss = min(avg_test_loss, min_test_loss)
                max_test_acc = max(avg_test_acc, max_test_acc)
                
                model.train()
