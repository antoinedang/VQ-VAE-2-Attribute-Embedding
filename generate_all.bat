@REM python3 .\vq-vae-2\sample_with_prior.py classical_vanilla --genre classical --checkpoint-folder checkpoints_vanilla --temp 0.01

@REM python3 .\vq-vae-2\sample_with_prior.py classical_0.1 --genre classical --checkpoint-folder checkpoints_0.1 --temp 0.01

python3 .\vq-vae-2\sample_with_bottom_prior.py classical_0.1_bottom --genre classical --checkpoint-folder checkpoints_0.1 --embeddings latent_embeddings_0.1 --temp 0.01

@REM python3 .\vq-vae-2\sample_with_bottom_prior.py classical_bottom_vanilla --genre classical --checkpoint-folder checkpoints_vanilla --embeddings latent_embeddings_vanilla --temp 0.01

python3 .\vq-vae-2\sample_with_top_prior.py classical_0.1_top --genre classical --checkpoint-folder checkpoints_0.1 --embeddings latent_embeddings_0.1 --temp 0.01

@REM python3 .\vq-vae-2\sample_with_top_prior.py classical_topvanilla --genre classical --checkpoint-folder checkpoints_vanilla --embeddings latent_embeddings_vanilla --temp 0.01