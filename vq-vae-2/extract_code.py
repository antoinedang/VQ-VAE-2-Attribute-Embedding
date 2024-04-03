import argparse
import pickle

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import lmdb
from tqdm import tqdm

from dataset import CodeRow
from model_definitions import getVQVAE

from tiff_dataset import TIFFDataset


def extract(lmdb_env, loader, model, device):
    index = 0

    with lmdb_env.begin(write=True) as txn:
        pbar = tqdm(loader)

        for img, label, filename in pbar:
            img = img.to(device)
            label = label.to(device)

            _, _, _, _, id_t, id_b = model.encode(img, label)
            id_t = id_t.detach().cpu().numpy()
            id_b = id_b.detach().cpu().numpy()
            label = label.detach().cpu().numpy()

            for file, label, top, bottom in zip(filename, label, id_t, id_b):
                row = CodeRow(label=label, top=top, bottom=bottom, filename=file)
                txn.put(str(index).encode('utf-8'), pickle.dumps(row))
                index += 1
                pbar.set_description(f'inserted: {index}')

        txn.put('length'.encode('utf-8'), str(index).encode('utf-8'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str)
    parser.add_argument('--name', type=str, default="latent_embeddings")
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('path', type=str)

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu": print("WARN: CUDA not available. Training will take very long.")

    dataset = TIFFDataset(args.path, provide_filename=True)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    model = getVQVAE(embed_labels=False, device=device)
    
    model.load_state_dict(torch.load(args.ckpt))
    model = model.to(device)
    model.eval()

    map_size = 100 * 1024 * 1024 * 1024

    env = lmdb.open(args.name, map_size=map_size)

    extract(env, loader, model, device)
