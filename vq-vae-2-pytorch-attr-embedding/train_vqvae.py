import argparse
import sys
import os

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from torchvision import datasets, transforms, utils

from tqdm import tqdm

from vqvae import VQVAE
from scheduler import CycleScheduler
import distributed as dist

from PIL import Image

from tiff_dataset import TIFFDataset, MAX_POSSIBLE_SPECTROGRAM_VALUE

def train(epoch, loader, model, optimizer, scheduler, device):
    if dist.is_primary():
        loader = tqdm(loader)

    criterion = nn.MSELoss()

    latent_loss_weight = 0.25
    attr_embedding_loss_weight = 0.1

    mse_sum = 0
    mse_n = 0

    for i, (img, label) in enumerate(loader):
        model.zero_grad()

        img = img.to(device)

        out, latent_loss, attr_embedding_loss = model(img, label)
        recon_loss = criterion(out, img)
        latent_loss = latent_loss.mean()
        loss = recon_loss + latent_loss_weight * latent_loss + attr_embedding_loss * attr_embedding_loss_weight
        loss.backward()

        if scheduler is not None:
            scheduler.step()
        optimizer.step()

        part_mse_sum = recon_loss.item() * img.shape[0]
        part_mse_n = img.shape[0]
        comm = {"mse_sum": part_mse_sum, "mse_n": part_mse_n}
        comm = dist.all_gather(comm)

        for part in comm:
            mse_sum += part["mse_sum"]
            mse_n += part["mse_n"]

        if dist.is_primary():
            lr = optimizer.param_groups[0]["lr"]

            loader.set_description(
                (
                    f"epoch: {epoch + 1}; mse: {recon_loss.item():.5f}; "
                    f"latent: {latent_loss.item():.5f}; attribute: {attr_embedding_loss.item() * attr_embedding_loss_weight:.5f}; avg mse: {mse_sum / mse_n:.5f}; "
                    f"lr: {lr:.5f}"
                )
            )

            if i % 100 == 0:
                model.eval()

                random_idx = torch.randint(0, img.shape[0], size=(1,))
                sample = img[random_idx]
                sample_label = label[random_idx]

                with torch.no_grad():
                    out, _, _ = model(sample, sample_label)
                
                side_by_side_img = (torch.hstack([torch.squeeze(sample), torch.squeeze(out)]).cpu().numpy() + 0.5) * MAX_POSSIBLE_SPECTROGRAM_VALUE

                img = Image.fromarray(side_by_side_img)
                img.save(f"{args.eval_sample_folder}/{str(epoch + 1).zfill(5)}_{str(i).zfill(5)}.tiff", format='TIFF')

                model.train()


def main(args):
    device = args.device

    args.distributed = dist.get_world_size() > 1

    dataset = TIFFDataset(args.path)
    sampler = dist.data_sampler(dataset, shuffle=True, distributed=args.distributed)
    loader = DataLoader(
        dataset, batch_size = args.batch_size // args.n_gpu, sampler=sampler, num_workers=2
    )

    model = VQVAE(device=device).to(device)

    if args.distributed:
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[dist.get_local_rank()],
            output_device=dist.get_local_rank(),
        )

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = None
    if args.sched == "cycle":
        scheduler = CycleScheduler(
            optimizer,
            args.lr,
            n_iter=len(loader) * args.epoch,
            momentum=None,
            warmup_proportion=0.05,
        )

    for i in range(args.epoch):
        train(i, loader, model, optimizer, scheduler, device)

        if dist.is_primary():
            torch.save(model.state_dict(), f"{args.checkpoint_folder}/vqvae_{str(i + 1).zfill(3)}.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_gpu", type=int, default=1)

    port = (
        2 ** 15
        + 2 ** 14
        + hash(os.getuid() if sys.platform != "win32" else 1) % 2 ** 14
    )
    parser.add_argument("--dist_url", default=f"tcp://127.0.0.1:{port}")

    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--epoch", type=int, default=560)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--sched", type=str)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--eval-sample-folder", type=str, default="eval_samples_attr_embedding")
    parser.add_argument("--checkpoint-folder", type=str, default="checkpoints_attr_embedding")
    parser.add_argument("path", type=str)

    args = parser.parse_args()

    print(args)

    dist.launch(main, args.n_gpu, 1, 0, args.dist_url, args=(args,))
