import torch

from torch import nn, Tensor


class AutoEncoderGarcia2020(nn.Module):
    def __init__(self):
        """ Autoencoder from Garcia et al. (2020) """        
        super().__init__()

        # Autoencoder from Garcia et al. (2020)
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=64, kernel_size=2, stride=2),
            nn.LeakyReLU(0.3),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=2, stride=2),
            nn.LeakyReLU(0.3),
            nn.Flatten(),
            nn.Linear(in_features=32768, out_features=2048), # Activation!
            nn.LeakyReLU(0.3),
            nn.Linear(in_features=2048, out_features=32768),
            nn.LeakyReLU(0.3),
            nn.Unflatten(dim=1, unflattened_size=(128, 16, 16)),
            nn.LeakyReLU(0.3),
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2),
            nn.LeakyReLU(0.3),
            nn.ConvTranspose2d(in_channels=64, out_channels=8, kernel_size=2, stride=2),
            nn.Sigmoid()
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): Single slices corresponding to segment of length 120
                Shape (N, 1, 64, 64) 

        Returns:
            x_reconstr: Reconstructed image(s) of shape (N, 1, 64, 64)
        """   
        return self.layers(x)     