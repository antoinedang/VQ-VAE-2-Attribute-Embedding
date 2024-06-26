# AUGMENT DATA SET
python3 augment_data.py


# for vanilla VQ-VAE 2
python3 ./vq-vae-2/train_vqvae.py AugmentedGTZAN/
# for custom VQ-VAE 2 with margin loss
python3 ./vq-vae-2/train_vqvae.py AugmentedGTZAN/ --use-attr-embedding


# ONCE ABOVE STEP IS COMPLETE, WE EXTRACT THE LATENT EMBEDDINGS OF THE MODEL
python3 ./vq-vae-2/extract_code.py --ckpt checkpoints/vqvae_[EPOCH].pt AugmentedGTZAN/


# ONCE LATENT EMBEDDINGS ARE EXTRACTED, WE TRAIN A PIXELSNAIL MODEL ON THE LATENT EMBEDDINGS
python3 ./vq-vae-2/train_pixelsnail.py latent_embeddings


# ONCE THE PIXELSNAIL MODELS ARE TRAINED, GENERATE NEW SAMPLES WITH
python3 ./vq-vae-2/sample.py reggae_sample_0.1 --genre reggae --batch 5 --checkpoint-folder checkpoints_0.1