import numpy as np
from tqdm import tqdm
import torch
from model import models_mae
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import os
from sklearn.cluster import KMeans
import umap
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image


class MAE_HIC:
  def __init__(self, model_path, data_path):
    self.model_path = model_path
    self.data_path = data_path

  def load_model(self, model = 'mae_vit_large_patch16'):
    model = models_mae.__dict__[model](norm_pix_loss=True)
    print("Loading model architecture")
    state_dict = torch.load(self.model_path, map_location=torch.device('cpu'))['model']
    print("Loading state_dict")
    model.load_state_dict(state_dict)
    print('State dict loaded')
    model.eval()
    return model

  def get_embeddings(self, dataloader, model):
    # not masking
    mask_ratio = 0
    data_size = len(dataloader.dataset)
    batch_size = dataloader.batch_size
    embedding_tensor = np.zeros((data_size, 196, 1024)).astype(np.float16)
    # Disable gradient computation for efficiency
    with torch.no_grad():
      for batch_idx, batch in enumerate(dataloader):
        imgs, labels = batch
        temp_size = imgs.shape[0]
        data_idx = batch_idx * batch_size
        # Forward pass through the encoder of the model
        latent, mask, ids_restore = model.forward_encoder(imgs, mask_ratio)
        # Exclude the class token from the latent representation
        latent_no_cls = latent[:, 1:, :]
        # Restore the latent representation using the gathered indices
        restored_latent = torch.gather(latent_no_cls, dim = 1, index = ids_restore.unsqueeze(-1).repeat(1, 1, latent.shape[2])).detach().cpu().numpy()
        embedding_tensor[data_idx:data_idx + temp_size] = restored_latent
    return embedding_tensor

  def load_dataset(self, batch_size=32, num_workers=20):
    transform_train = transforms.Compose([
          transforms.ToTensor()
          ])
    dataset_train = datasets.ImageFolder(os.path.join(self.data_path, 'train'), transform=transform_train)
    data_loader_train = torch.utils.data.DataLoader(
      dataset_train,
      batch_size=batch_size,
      num_workers=num_workers,
      drop_last=False
    )
    return data_loader_train


def main():
  model_path = '/gpfs/data/abl/home/shenkn01/MAE/mae/output_dir/checkpoint-799.pth'
  data_path = '/gpfs/scratch/shenkn01/mae_input'

  mae_hic = MAE_HIC(model_path, data_path)

  # Generate embeddings
  dataloader = mae_hic.load_dataset()
  model = mae_hic.load_model() 
  embeddings = mae_hic.get_embeddings(dataloader, model)

  b, f1, f2 = embeddings.shape
  embeddings = embeddings.reshape(b, f1 * f2)
  print(embeddings.shape)
  np.save('IMR90_embedding_2D_notransform.npy', embeddings)

if __name__ == "__main__":
      main()
