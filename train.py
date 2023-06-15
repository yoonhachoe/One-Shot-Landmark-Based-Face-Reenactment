import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import transforms
from torchsummary import summary
import argparse
from typing import Tuple
import dnnlib
from glob import glob
from tqdm import tqdm
import numpy as np
import legacy
import mapping_net
from dataset import MyDataset


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

def train(args):
    device = 'cuda:2' if torch.cuda.is_available() else 'cpu'

    writer_train = SummaryWriter(log_dir="tensorboard_logs/mapping/train")
    writer_val = SummaryWriter(log_dir="tensorboard_logs/mapping/val")

    transform = transforms.Compose([transforms.Resize(580), transforms.ToTensor()])

    LANDMARK_PATH_LIST = sorted(glob('face_landmark/*.png'))
    LANDMARK_ID_PATH_LIST = sorted(glob('face_landmark_id/*.png'))
    LATENT_PATH_PKL = 'latents_MEAD.pkl'
    LATENT_ID_PATH_PKL = 'latents_MEAD_ID.pkl'
    dataset = MyDataset(LANDMARK_PATH_LIST, LANDMARK_ID_PATH_LIST, LATENT_ID_PATH_PKL, LATENT_PATH_PKL, transform, device)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    trainloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    testloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    # Network
    network_pkl = "https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-ffhq-1024x1024.pkl"
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)  # type: ignore

    if hasattr(G.synthesis, 'input'):
        m = make_transform(translate=(0, 0), angle=0)
        m = np.linalg.inv(m)
        G.synthesis.input.transform.copy_(torch.from_numpy(m))

    # Model
    landmark_encoder = mapping_net.LandmarkEncoder()
    model = mapping_net.MappingNetwork(landmark_encoder).to(device)
    summary(model, [(3, 580, 580), (3, 580, 580)])

    # Loss
    MSE_loss = nn.MSELoss(reduction="mean").to(device)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=2e-4, betas=(0.9, 0.999), weight_decay=1e-4)

    pbar = tqdm(range(args.epochs))

    for e in pbar:
        optimizer.param_groups[0]["lr"] = 2e-3
        sum_train_loss = 0.0
        sum_val_loss = 0.0
        for i, item in enumerate(trainloader):
            optimizer.zero_grad()
            model.train()
            landmark_id, landmark, latent_id, latent = item
            landmark_id = landmark_id.to(device)
            landmark = landmark.to(device)
            latent = latent.flatten(1).to(device)  # batch * 8192
            latent_id = latent_id.flatten(1).to(device)  # batch * 8192
            pred_delta = model.forward(landmark_id, landmark)
            pred = latent_id + pred_delta
            train_mse_loss = MSE_loss(pred, latent)
            train_loss = args.mse_lambda * train_mse_loss
            train_loss.backward()
            sum_train_loss += train_loss.item()
            optimizer.step()

        writer_train.add_scalar('ex1', sum_train_loss / len(trainloader), e)

        for i, item in enumerate(testloader):
            model.eval()
            landmark_id, landmark, latent_id, latent = item
            landmark_id = landmark_id.to(device)
            landmark = landmark.to(device)
            latent = latent.flatten(1).to(device)  # batch * 8192
            latent_id = latent_id.flatten(1).to(device)  # batch * 8192
            with torch.no_grad():
                pred_delta = model.forward(landmark_id, landmark)
                pred = latent_id + pred_delta
            val_mse_loss = MSE_loss(pred, latent)
            val_loss = args.mse_lambda * val_mse_loss
            sum_val_loss += val_loss.item()

        writer_val.add_scalar('ex1', sum_val_loss / len(testloader), e)
        pbar.set_description(f"train loss: {train_loss.item():.4f}, val loss: {val_loss.item():.4f};")
        torch.save(model.state_dict(), 'mapping_model/model_ex1_{epoch:03d}epochs.pth'.format(epoch=e + 1))

# ---------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default="20", help="Number of epochs to train")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for train and test loader")
    parser.add_argument("--mse_lambda", type=float, default=10, help="weight of mse loss")

    args = parser.parse_args()
    train(args)
