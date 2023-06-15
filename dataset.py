from torch.utils.data import Dataset
import torch
import pickle
from PIL import Image


class MyDataset(Dataset):
    def __init__(self, landmark_path_list, landmark_id_path_list, latent_id_path_pkl, latent_path_pkl, transform, device):
        self.landmark_path_list = landmark_path_list
        self.landmark_id_path_list = landmark_id_path_list
        with open(latent_id_path_pkl, 'rb') as f:
            self.latent_id = pickle.load(f)
        with open(latent_path_pkl, 'rb') as f:
            self.latent = pickle.load(f)
        self.transform = transform
        self.device = device

    def __len__(self):
        return len(self.landmark_path_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        landmark = Image.open(self.landmark_path_list[idx])
        landmark = landmark.convert("RGB")
        if self.transform is not None:
            landmark = self.transform(landmark)

        if idx >= 0 and idx < 3382:
            landmark_id = Image.open(self.landmark_id_path_list[0])
        elif idx >= 3382 and idx < 7413:
            landmark_id = Image.open(self.landmark_id_path_list[1])
        elif idx >= 7413 and idx < 11112:
            landmark_id = Image.open(self.landmark_id_path_list[2])
        elif idx >= 11112 and idx < 14863:
            landmark_id = Image.open(self.landmark_id_path_list[3])
        elif idx >= 14863 and idx < 19064:
            landmark_id = Image.open(self.landmark_id_path_list[4])
        elif idx >= 19064 and idx < 23141:
            landmark_id = Image.open(self.landmark_id_path_list[5])
        elif idx >= 23141 and idx < 28268:
            landmark_id = Image.open(self.landmark_id_path_list[6])
        elif idx >= 28268 and idx < 32390:
            landmark_id = Image.open(self.landmark_id_path_list[7])
        elif idx >= 32390 and idx < 36862:
            landmark_id = Image.open(self.landmark_id_path_list[8])
        elif idx >= 36862 and idx < 40683:
            landmark_id = Image.open(self.landmark_id_path_list[9])
        elif idx >= 40683 and idx < 45264:
            landmark_id = Image.open(self.landmark_id_path_list[10])
        elif idx >= 45264 and idx < 50203:
            landmark_id = Image.open(self.landmark_id_path_list[11])
        elif idx >= 50203 and idx < 55143:
            landmark_id = Image.open(self.landmark_id_path_list[12])
        elif idx >= 55143 and idx < 59407:
            landmark_id = Image.open(self.landmark_id_path_list[13])
        elif idx >= 59407 and idx < 63477:
            landmark_id = Image.open(self.landmark_id_path_list[14])
        elif idx >= 63477 and idx < 69272:
            landmark_id = Image.open(self.landmark_id_path_list[15])
        elif idx >= 69272 and idx < 73661:
            landmark_id = Image.open(self.landmark_id_path_list[16])
        elif idx >= 73661 and idx < 77921:
            landmark_id = Image.open(self.landmark_id_path_list[17])
        elif idx >= 77921 and idx < 83216:
            landmark_id = Image.open(self.landmark_id_path_list[18])
        elif idx >= 83216 and idx < 88109:
            landmark_id = Image.open(self.landmark_id_path_list[19])
        elif idx >= 88109 and idx < 92266:
            landmark_id = Image.open(self.landmark_id_path_list[10])
        elif idx >= 92266 and idx < 95926:
            landmark_id = Image.open(self.landmark_id_path_list[21])
        elif idx >= 95926 and idx < 100063:
            landmark_id = Image.open(self.landmark_id_path_list[22])
        elif idx >= 100063 and idx < 102995:
            landmark_id = Image.open(self.landmark_id_path_list[23])
        elif idx >= 102995 and idx < 107364:
            landmark_id = Image.open(self.landmark_id_path_list[24])
        elif idx >= 107364 and idx < 111839:
            landmark_id = Image.open(self.landmark_id_path_list[25])
        elif idx >= 111839 and idx < 116134:
            landmark_id = Image.open(self.landmark_id_path_list[26])
        elif idx >= 116134 and idx < 121201:
            landmark_id = Image.open(self.landmark_id_path_list[27])
        elif idx >= 121201 and idx < 125971:
            landmark_id = Image.open(self.landmark_id_path_list[28])
        elif idx >= 125971 and idx < 130289:
            landmark_id = Image.open(self.landmark_id_path_list[29])
        elif idx >= 130289 and idx < 134649:
            landmark_id = Image.open(self.landmark_id_path_list[30])
        elif idx >= 134649 and idx < 140499:
            landmark_id = Image.open(self.landmark_id_path_list[31])
        elif idx >= 140499 and idx < 145204:
            landmark_id = Image.open(self.landmark_id_path_list[32])
        elif idx >= 145204 and idx < 149158:
            landmark_id = Image.open(self.landmark_id_path_list[33])
        elif idx >= 149158 and idx < 153109:
            landmark_id = Image.open(self.landmark_id_path_list[34])
        elif idx >= 153109 and idx < 156780:
            landmark_id = Image.open(self.landmark_id_path_list[35])
        elif idx >= 156780 and idx < 161784:
            landmark_id = Image.open(self.landmark_id_path_list[36])
        elif idx >= 161784 and idx < 165797:
            landmark_id = Image.open(self.landmark_id_path_list[37])
        elif idx >= 165797 and idx < 170113:
            landmark_id = Image.open(self.landmark_id_path_list[38])
        elif idx >= 170113 and idx < 174021:
            landmark_id = Image.open(self.landmark_id_path_list[39])
        elif idx >= 174021 and idx < 177927:
            landmark_id = Image.open(self.landmark_id_path_list[40])
        elif idx >= 177927 and idx < 181836:
            landmark_id = Image.open(self.landmark_id_path_list[41])
        landmark_id = landmark_id.convert("RGB")
        if self.transform is not None:
            landmark_id = self.transform(landmark_id)

        latent_id = self.latent_id[idx].to(self.device)
        latent = self.latent[idx].to(self.device)

        return landmark_id, landmark, latent_id, latent

