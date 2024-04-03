# for vanilla VQ-VAE 2
python3 ./vq-vae-2/train_vqvae.py AugmentedGTZAN/ --size 512 --batch_size 8 --eval-sample-interval 500

# for custom VQ-VAE 2 with triplet loss
python3 ./vq-vae-2/train_vqvae.py AugmentedGTZAN/ --size 512 --batch_size 8 --use-attr-embedding --eval-sample-interval 500