import numpy as np
import torch
from model import models_mae
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import os
from sklearn.cluster import KMeans
import umap
import seaborn as sns

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
      # Disable gradient computation for efficiency
      with torch.no_grad():
        for batch in dataloader:
          imgs, labels = batch
          # Forward pass through the encoder of the model
          latent, mask, ids_restore = model.forward_encoder(imgs, mask_ratio)
          # Exclude the class token from the latent representation
          latent_no_cls = latent[:, 1:, :]
          # Restore the latent representation using the gathered indices
          restored_latent = torch.gather(latent_no_cls, dim = 1, index = ids_restore.unsqueeze(-1).repeat(1, 1, latent.shape[2])).detach().cpu().numpy()
          return restored_latent
  
  def load_dataset(self, batch_size=32, num_workers=20):
      transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
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
  embeddings = embeddings.reshape(-1, 2)


  # Create UMAP
  umap_embedding = umap.UMAP().fit_transform(embeddings)

  # Plot UMAP embedding with cluster colors
  plt.figure(figsize=(10, 8))
  if n_clusters <= 20:
      palette = 'tab20'
  else:
      palette = 'Spectral'
  sns.scatterplot(x=umap_embedding[:, 0], y=umap_embedding[:, 1], hue=kmeans_labels, palette=palette, legend='full')
  plt.title(f'K-means Clustering with UMAP Visualization')
  plt.legend(bbox_to_anchor=(1.1, 1.05))
  plt.savefig('umap_visualization.png')
  plt.show()

if __name__ == "__main__":
      main()
