# for vanilla VQ-VAE 2
python3 ./vq-vae-2-pytorch-vanilla/train_vqvae.py AugmentedGTZAN/ --size 512 --batch_size 32

# for custom VQ-VAE 2 with triplet loss
python3 ./vq-vae-2-pytorch-attr-embedding/train_vqvae.py AugmentedGTZAN/ --size 512 --batch_size 32