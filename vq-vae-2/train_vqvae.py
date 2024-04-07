import argparse
import sys
import os

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from tqdm import tqdm

from scheduler import CycleScheduler
from model_definitions import getVQVAE
import distributed as dist

from PIL import Image

from tiff_dataset import TIFFDataset
import numpy as np

def train(epoch, train_loader, test_set, model, optimizer, scheduler, device, eval_sample_interval, attr_loss_weight):
    if dist.is_primary():
        train_loader = tqdm(train_loader)

    criterion = nn.MSELoss()

    latent_loss_weight = 0.25
    attr_embedding_loss_weight = attr_loss_weight

    mse_sum = 0
    mse_n = 0

    for i, (img, label) in enumerate(train_loader):
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

            train_loader.set_description(
                (
                    f"epoch: {epoch + 1}; mse: {recon_loss.item():.5f}; "
                    f"latent: {latent_loss.item():.5f}; attribute: {attr_embedding_loss.item() * attr_embedding_loss_weight:.5f}; avg mse: {mse_sum / mse_n:.5f}; "
                    f"lr: {lr:.5f}"
                )
            )

            if i % eval_sample_interval == 0:
                model.eval()
    
                # TRAIN EVAL
    
                random_idx = torch.randint(0, img.shape[0], size=(1,))
                sample = img[random_idx]
                sample_label = label[random_idx]

                with torch.no_grad():
                    out, _, _ = model(sample, sample_label)
                
                side_by_side_img = np.exp(torch.hstack([torch.squeeze(sample), torch.squeeze(out)]).cpu().numpy()) - 1.0

                img = Image.fromarray(side_by_side_img)
                img.save(f"{args.eval_sample_folder}/{str(epoch + 1).zfill(5)}_{str(i).zfill(5)}_train.tiff", format='TIFF')

                # TEST EVAL
                
                random_idx = torch.randint(0, len(test_set), size=(1,))
                sample, sample_label = test_set[random_idx]
                
                sample = sample.to(device).unsqueeze(0)

                with torch.no_grad():
                    out, _, _ = model(sample, sample_label)
                
                side_by_side_img = np.exp(torch.hstack([torch.squeeze(sample), torch.squeeze(out)]).cpu().numpy()) - 1.0

                img = Image.fromarray(side_by_side_img)
                img.save(f"{args.eval_sample_folder}/{str(epoch + 1).zfill(5)}_{str(i).zfill(5)}_test.tiff", format='TIFF')

                model.train()


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu": print("WARN: CUDA not available. Training will take very long.")

    args.distributed = dist.get_world_size() > 1

    dataset = TIFFDataset(args.path)
    train_size = int(0.9 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_sampler = dist.data_sampler(train_dataset, shuffle=True, distributed=args.distributed)
    train_loader = DataLoader(
        train_dataset, batch_size = args.batch_size // args.n_gpu, sampler=train_sampler, num_workers=2
    )

    model = getVQVAE(embed_labels=args.use_attr_embedding, device=device)

    if args.distributed:
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[dist.get_local_rank()],
            output_device=dist.get_local_rank(),
        )

    if args.checkpoint is not None:
        model.load_state_dict(torch.load(args.checkpoint))
        ckpt_filename = os.path.basename(args.checkpoint)
        current_epochs = int(ckpt_filename.split("_")[1].split(".")[0])
    else:
        current_epochs = 0
        
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for i in range(args.epoch):
        train(current_epochs + i, train_loader, test_dataset, model, optimizer, None, device, args.eval_sample_interval, args.attr_loss_weight)

        if dist.is_primary() and current_epochs + i == args.epoch - 1:
            torch.save(model.state_dict(), f"{args.checkpoint_folder}/vqvae_{str(current_epochs + i + 1).zfill(3)}.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_gpu", type=int, default=1)

    port = (
        2 ** 15
        + 2 ** 14
        + hash(os.getuid() if sys.platform != "win32" else 1) % 2 ** 14
    )
    parser.add_argument("--dist_url", default=f"tcp://127.0.0.1:{port}")

    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--epoch", type=int, default=50)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--eval-sample-folder", type=str, default="eval_samples")
    parser.add_argument("--checkpoint-folder", type=str, default="checkpoints")
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--eval-sample-interval", type=int, default=500)
    parser.add_argument("--use-attr-embedding", action='store_true')
    parser.add_argument("--attr-loss-weight", type=float)
    parser.add_argument("path", type=str)

    args = parser.parse_args()

    print(args)

    dist.launch(main, args.n_gpu, 1, 0, args.dist_url, args=(args,))
