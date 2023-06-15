import torch
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
from typing import Tuple
import dnnlib
from glob import glob
import numpy as np
import pickle
import re
import legacy
import mapping_net


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

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
landmark_encoder = mapping_net.LandmarkEncoder()
model = mapping_net.MappingNetwork(landmark_encoder)
model.load_state_dict(torch.load("mapping_model/model_ex1_020epochs.pth"))
model.eval().to(device)
print(model)

network_pkl = "https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-ffhq-1024x1024.pkl"
with dnnlib.util.open_url(network_pkl) as f:
    G = legacy.load_network_pkl(f)['G_ema'].to(device)  # type: ignore

if hasattr(G.synthesis, 'input'):
    m = make_transform(translate=(0, 0), angle=0)
    m = np.linalg.inv(m)
    G.synthesis.input.transform.copy_(torch.from_numpy(m))

imglist = sorted(glob('face_landmark_test/*.png'))
imglist_id = sorted(glob('face_landmark_id_test/*.png'))
with open('latents_ID_test.pkl', 'rb') as f:
    latent_id = pickle.load(f)
latent_id = latent_id[0].flatten(0).unsqueeze(0).to(device)
for img in imglist:
    image = Image.open(img)
    image = image.convert("RGB")
    transform = transforms.Compose([transforms.Resize(580), transforms.ToTensor()])
    landmark = transform(image).unsqueeze(0).to(device)
    image_id = Image.open(imglist_id[0])  # change the source identity with index
    imag_id = image_id.convert("RGB")
    transform = transforms.Compose([transforms.Resize(580), transforms.ToTensor()])
    landmark_id = transform(image_id).unsqueeze(0).to(device)
    with torch.no_grad():
        pred_delta = model.forward(landmark_id, landmark)  # batch * 8192
        pred = latent_id + pred_delta
        syn_pred = G.synthesis(pred.reshape(1, 16, 512))
        syn_pred = (syn_pred + 1.0) / 2.0
    save_image(syn_pred, f"test_result/{re.split('[/ .]', img)[1]}.jpg")