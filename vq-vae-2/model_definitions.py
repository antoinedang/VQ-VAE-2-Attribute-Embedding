from vqvae import VQVAE
from pixelsnail import PixelSNAIL

def getVQVAE(embed_labels, device):
    return VQVAE(
        embed_labels=embed_labels,
        in_channel=1,
        channel=128,
        n_res_block=2*2, # * 2 because these parameters were for 256x256 image, we are now doing 512x512
        n_res_channel=32*2,
        embed_dim=64*2,
        n_embed=512*2,
        device=device).to(device)
    
def getPixelSnailBottom():
    return PixelSNAIL(
        shape=[64*2, 64*2],
        n_class=1024,
        kernel_size=5,
        attention=False,
        dropout=0.1,
        channel=int(256 * 2),
        n_block=int(4 * 0.5),
        res_channel=int(256 * 2),
        n_res_block=int(4 * 0.5),
        cond_res_channel=int(256 * 2),
        n_cond_res_block=int(4 * 0.5),
    )
    

def getPixelSnailTop():
    return PixelSNAIL(
        shape=[32*2, 32*2],
        n_class=1024,
        kernel_size=5,
        dropout=0.1,
        n_out_res_block=0,
        channel=int(256 * 2),
        n_block=int(4 * 0.5),
        n_res_block=int(4 * 0.5),
        res_channel=int(256 * 2),
    )