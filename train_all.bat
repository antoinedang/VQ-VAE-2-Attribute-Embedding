@REM AUGMENT DATASET
@REM python augment_data.py

@REM VANILLA VQVAE
@REM python ./vq-vae-2/train_vqvae.py ProcessedGTZAN/ --eval-sample-folder eval_samples_vanilla --checkpoint-folder checkpoints_vanilla

@REM VANILLA AUGMENTED VQVAE
@REM python ./vq-vae-2/train_vqvae.py AugmentedGTZAN/ --eval-sample-folder eval_samples_vanilla_augmented --checkpoint-folder checkpoints_vanilla_augmented

@REM @REM 0.1 WEIGHTING
@REM python ./vq-vae-2/train_vqvae.py ProcessedGTZAN/ --use-attr-embedding --eval-sample-folder eval_samples_0.1 --checkpoint-folder checkpoints_0.1 --attr-loss-weight 0.1

@REM @REM 1 WEIGHTING
@REM python ./vq-vae-2/train_vqvae.py ProcessedGTZAN/ --use-attr-embedding --eval-sample-folder eval_samples_1 --checkpoint-folder checkpoints_1 --attr-loss-weight 1

@REM @REM 10 WEIGHTING
@REM python ./vq-vae-2/train_vqvae.py ProcessedGTZAN/ --use-attr-embedding --eval-sample-folder eval_samples_10 --checkpoint-folder checkpoints_10 --attr-loss-weight 10

@REM VANILLA CODE EXTRACT + PIXELCNN
@REM python ./vq-vae-2/extract_code.py ProcessedGTZAN/ --ckpt checkpoints_vanilla/vqvae_200.pt --name latent_embeddings_vanilla
@REM python ./vq-vae-2/train_pixelsnail.py latent_embeddings_vanilla --checkpoint-folder checkpoints_vanilla

@REM VANILLA AUGMENTED CODE EXTRACT + PIXELCNN
@REM python ./vq-vae-2/extract_code.py AugmentedGTZAN/ --ckpt checkpoints_vanilla_augmented/vqvae_200.pt --name latent_embeddings_vanilla_augmented
@REM python ./vq-vae-2/train_pixelsnail.py latent_embeddings_vanilla_augmented --checkpoint-folder checkpoints_vanilla_augmented

@REM WEIGHTING 0.1 CODE EXTRACT + PIXELCNN
@REM python ./vq-vae-2/extract_code.py ProcessedGTZAN/ --ckpt checkpoints_0.1/vqvae_200.pt --name latent_embeddings_0.1
python ./vq-vae-2/train_pixelsnail.py latent_embeddings_0.1 --checkpoint-folder checkpoints_0.1

@REM WEIGHTING 1 CODE EXTRACT + PIXELCNN
@REM python ./vq-vae-2/extract_code.py ProcessedGTZAN/ --ckpt checkpoints_1/vqvae_200.pt --name latent_embeddings_1
python ./vq-vae-2/train_pixelsnail.py latent_embeddings_1 --checkpoint-folder checkpoints_1

@REM WEIGHTING 10 CODE EXTRACT + PIXELCNN
@REM python ./vq-vae-2/extract_code.py ProcessedGTZAN/ --ckpt checkpoints_10/vqvae_200.pt --name latent_embeddings_10
python ./vq-vae-2/train_pixelsnail.py latent_embeddings_10 --checkpoint-folder checkpoints_10


