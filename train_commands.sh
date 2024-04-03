# AUGMENT DATA SET
python3 augment_data.py


# for vanilla VQ-VAE 2
python3 ./vq-vae-2/train_vqvae.py AugmentedGTZAN/
# for custom VQ-VAE 2 with triplet loss
python3 ./vq-vae-2/train_vqvae.py AugmentedGTZAN/ --use-attr-embedding


# ONCE ABOVE STEP IS COMPLETE, WE EXTRACT THE LATENT EMBEDDINGS OF THE MODEL
python3 ./vq-vae-2/extract_code.py --ckpt checkpoints/vqvae_045.pt AugmentedGTZAN/


# ONCE LATENT EMBEDDINGS ARE EXTRACTED, WE TRAIN A PIXELSNAIL MODEL ON THE LATENT EMBEDDINGS
python3 ./vq-vae-2/train_pixelsnail.py latent_embeddings


# ONCE THE PIXELSNAIL MODELS ARE TRAINED, GENERATE NEW SAMPLES WITH
python3 ./vq-vae-2/sample.py blues_sample.wav --genre blues --batch 1