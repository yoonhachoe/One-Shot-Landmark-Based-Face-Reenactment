import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
import argparse
from typing import Tuple
import dnnlib
from glob import glob
from tqdm import tqdm
import numpy as np
import legacy
import pickle
import lpips
from id_loss import IDLoss


def make_transform(translate: Tuple[float,float], angle: float):
    m = np.eye(3)
    s = np.sin(angle/360.0*np.pi*2)
    c = np.cos(angle/360.0*np.pi*2)
    m[0][0] = c
    m[0][1] = s
    m[0][2] = translate[0]
    m[1][0] = -s
    m[1][1] = c
    m[1][2] = translate[1]
    return m

#----------------------------------------------------------------------------

def main(args):
    device = 'cuda:2' if torch.cuda.is_available() else 'cpu'
    writer = SummaryWriter(log_dir=f"tensorboard_logs/opt")

    # Read an image
    imgdir = sorted(glob(args.img_dir + '/*'))
    latents_list = []

    for i in range(len(imgdir)):
        img_name = imgdir[i].split("/")[-1]
        image = Image.open(imgdir[i])
        image = image.convert("RGB")
        transform = transforms.Compose([transforms.Resize(1024), transforms.ToTensor()])  # normalize to [0, 1]
        image = transform(image)
        image = image.unsqueeze(0)
        image = image.to(device)

        # Network
        network_pkl = "https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-ffhq-1024x1024.pkl"
        with dnnlib.util.open_url(network_pkl) as f:
            G = legacy.load_network_pkl(f)['G_ema'].to(device)  # type: ignore

        if hasattr(G.synthesis, 'input'):
            m = make_transform(translate=(0, 0), angle=0)
            m = np.linalg.inv(m)
            G.synthesis.input.transform.copy_(torch.from_numpy(m))

        # Loss
        MSE_loss = nn.MSELoss(reduction="mean")
        loss_fn_alex = lpips.LPIPS(net='alex').to(device)
        id_loss = IDLoss(device)

        z_samples = np.random.RandomState(123).randn(10000, G.z_dim)
        w_samples = G.mapping(torch.from_numpy(z_samples).to(device), None)  # torch.Size([10000, 16, 512])
        latents_rand = torch.mean(w_samples, dim=0, keepdim=True)  # torch.Size([1, 16, 512])
        latents_rand.requires_grad = True

        optimizer = optim.Adam({latents_rand}, lr=0.01, betas=(0.9, 0.999), eps=1e-8)

        pbar = tqdm(range(args.step))

        # Loop to optimize latent vector to match the generated image to input image
        for e in pbar:
            optimizer.zero_grad()
            syn_img = G.synthesis(latents_rand)
            syn_img = (syn_img + 1.0) / 2.0  # normalize to [0, 1]
            mse = MSE_loss(syn_img, image)
            p = loss_fn_alex(syn_img, image)
            i_loss = id_loss(syn_img, image)[0]
            delta = latents_rand[:, 0, :] - latents_rand[:, 1, :]
            for i in range(2, 16):
                delta = torch.cat((delta, latents_rand[:, 0, :] - latents_rand[:, i, :]), dim=0)  # torch.Size([15, 512])
            reg = torch.norm(delta, p=2, dim=1).sum()
            loss = args.mse_lambda*mse + args.lpips_lambda*p + args.id_lambda*i_loss + args.reg_lambda*reg
            loss.backward()
            optimizer.step()
            loss_np = loss.detach().cpu().numpy()
            loss_np = loss_np[0, 0, 0, 0]
            writer.add_scalar('Optimization loss (MSE)', mse, e)
            writer.add_scalar('Optimization loss (LPIPS)', p, e)
            writer.add_scalar('Optimization loss (ID)', i_loss, e)
            writer.add_scalar('L2 regularizer', reg, e)
            writer.add_scalar('Optimization loss (Total)', loss_np, e)
            pbar.set_description(f"loss: {loss.item():.4f};")

            if e+1 == 1000:
                save_image(syn_img, f"./opt_MEAD/optimized_{img_name}")

        latents_rand_copy = latents_rand.detach().clone()
        latents_list.append(latents_rand_copy)

    with open('latents_MEAD.pkl', 'wb') as f:
        pickle.dump(latents_list, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir", type=str, required=True, help="the input image directory")
    parser.add_argument("--step", type=int, default=1000, help="the number of optimization steps")
    parser.add_argument("--mse_lambda", type=float, default=0.1, help="weight of mse loss")
    parser.add_argument("--lpips_lambda", type=float, default=1.5, help="weight of lpips loss")
    parser.add_argument("--id_lambda", type=float, default=0.05, help="weight of id loss")
    parser.add_argument("--reg_lambda", type=float, default=0, help="weight of delta regularizer")

    args = parser.parse_args()
    main(args)
