import torch
import torch.nn as nn
import torch.nn.functional as F

class ImgDecoder(nn.Module):
    def __init__(self, input_dim=1, latent_dim=64, with_logits=False):
        super(ImgDecoder, self).__init__()
        self.with_logits = with_logits
        self.n_channels = input_dim
        
        self.dense = nn.Linear(latent_dim, 512)
        self.dense1 = nn.Linear(512, 128 * 13 * 7)  # 13x7 是初始feature map大小
        
        self.deconv1 = nn.ConvTranspose2d(128, 128, kernel_size=3, stride=1, padding=1)  # 13x7 -> 13x7
        self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)   # 13x7 -> 26x14
        self.deconv3 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)    # 26x14 -> 52x28
        self.deconv4 = nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1)    # 52x28 -> 104x56
        self.deconv5 = nn.ConvTranspose2d(16, self.n_channels, kernel_size=4, stride=2, padding=1)  # 104x56 -> 208x112

    def forward(self, z):
        return self.decode(z)

    def decode(self, z):
        x = self.dense(z)
        x = torch.relu(x)
        x = self.dense1(x)
        x = x.view(x.size(0), 128, 13, 7)
        
        x = self.deconv1(x)
        x = torch.relu(x)
        
        x = self.deconv2(x)
        x = torch.relu(x)
        
        x = self.deconv3(x)
        x = torch.relu(x)
        
        x = self.deconv4(x)
        x = torch.relu(x)
        
        x = self.deconv5(x)
        
        # 输出大概是 (batch, 1, 208, 112)，下面插值调整到 (212, 120)
        if not self.with_logits:
            x = torch.sigmoid(x)
        x = F.interpolate(x, size=(120, 212), mode='bilinear', align_corners=False)
        
        return x



class ImgEncoder(nn.Module):
    """
    ResNet8 architecture as encoder.
    """

    def __init__(self, input_dim, latent_dim):
        """
        Parameters:
        ----------
        input_dim: int
            Number of input channels in the image.
        latent_dim: int
            Number of latent dimensions
        """
        super(ImgEncoder, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.define_encoder()
        self.elu = nn.ELU()
        print("Defined encoder.")

    def define_encoder(self):
        # define conv functions
        self.conv0 = nn.Conv2d(self.input_dim, 32, kernel_size=5, stride=2, padding=2)
        self.conv0_1 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=2)
        nn.init.xavier_uniform_(self.conv0_1.weight, gain=nn.init.calculate_gain("linear"))
        nn.init.zeros_(self.conv0_1.bias)

        self.conv1_0 = nn.Conv2d(32, 32, kernel_size=5, stride=2, padding=1)
        self.conv1_1 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        nn.init.xavier_uniform_(self.conv1_1.weight, gain=nn.init.calculate_gain("linear"))
        nn.init.zeros_(self.conv1_1.bias)

        self.conv2_0 = nn.Conv2d(64, 64, kernel_size=5, stride=2, padding=2)
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        nn.init.xavier_uniform_(self.conv2_1.weight, gain=nn.init.calculate_gain("linear"))
        nn.init.zeros_(self.conv2_1.bias)

        self.conv3_0 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        self.conv0_jump_2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.conv1_jump_3 = nn.Conv2d(64, 128, kernel_size=5, stride=4, padding=(2, 1))

        self.dense0 = nn.Linear(4 * 7 * 128, 512)
        self.dense1 = nn.Linear(512, 2 * self.latent_dim)

        print("Encoder network initialized.")

    def forward(self, img):
        return self.encode(img)
    
    def center_crop(self, tensor, target_tensor):
        """Crop tensor to match target_tensor shape."""
        _, _, h, w = tensor.shape
        _, _, th, tw = target_tensor.shape
        dh = (h - th) // 2
        dw = (w - tw) // 2
        return tensor[:, :, dh:dh+th, dw:dw+tw]

    def encode(self, img):
        """
        Encodes the input image.
        """

        # conv0
        x0_0 = self.conv0(img)
        x0_1 = self.conv0_1(x0_0)
        x0_1 = self.elu(x0_1)

        x1_0 = self.conv1_0(x0_1)
        x1_1 = self.conv1_1(x1_0)

        x0_jump_2 = self.conv0_jump_2(x0_1)

        x0_jump_2 = self.center_crop(x0_jump_2, x1_1) # Crop tensor to match target_tensor shape
        x1_1 = x1_1 + x0_jump_2

        x1_1 = self.elu(x1_1)

        x2_0 = self.conv2_0(x1_1)
        x2_1 = self.conv2_1(x2_0)

        x1_jump3 = self.conv1_jump_3(x1_1)

        x1_jump3 = self.center_crop(x1_jump3, x2_1)
        x2_1 = x2_1 + x1_jump3

        x2_1 = self.elu(x2_1)

        x3_0 = self.conv3_0(x2_1)

        x = x3_0.view(x3_0.size(0), -1)

        x = self.dense0(x)
        x = self.elu(x)
        x = self.dense1(x)
        return x


class Lambda(nn.Module):
    """Lambda function that accepts tensors as input."""

    def __init__(self, func):
        super(Lambda, self).__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)


class VAE(nn.Module):
    """Variational Autoencoder for reconstruction of depth images."""

    def __init__(self, input_dim=1, latent_dim=64, with_logits=False, inference_mode=False):
        """
        Parameters
        ----------
        input_dim: int
            The number of input channels in an image.
        latent_dim: int
            The latent dimension.
        """

        super(VAE, self).__init__()

        self.with_logits = with_logits
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.inference_mode = inference_mode
        self.encoder = ImgEncoder(input_dim=self.input_dim, latent_dim=self.latent_dim)
        self.img_decoder = ImgDecoder(
            input_dim=1, latent_dim=self.latent_dim, with_logits=self.with_logits
        )

        self.mean_params = Lambda(lambda x: x[:, : self.latent_dim])  # mean parameters
        self.logvar_params = Lambda(lambda x: x[:, self.latent_dim :])  # log variance parameters

    def forward(self, img):
        """Do a forward pass of the VAE. Generates a reconstructed image based on img
        Parameters
        ----------
        img: torch.Tensor
            The input image.
        """

        # encode
        z = self.encoder(img)

        # reparametrization trick
        mean = self.mean_params(z)
        logvar = self.logvar_params(z)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        if self.inference_mode:
            eps = torch.zeros_like(eps)
        z_sampled = mean + eps * std

        # decode
        img_recon = self.img_decoder(z_sampled)
        return img_recon, mean, logvar, z_sampled

    def forward_test(self, img):
        """Do a forward pass of the VAE. Generates a reconstructed image based on img
        Parameters
        ----------
        img: torch.Tensor
            The input image.
        """

        # encode
        z = self.encoder(img)

        # reparametrization trick
        mean = self.mean_params(z)
        logvar = self.logvar_params(z)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        if self.inference_mode:
            eps = torch.zeros_like(eps)
        z_sampled = mean + eps * std

        # decode
        img_recon = self.img_decoder(z_sampled)
        return img_recon, mean, logvar, z_sampled

    def encode(self, img):
        """Do a forward pass of the VAE. Generates a latent vector based on img
        Parameters
        ----------
        img: torch.Tensor
            The input image.
        """
        z = self.encoder(img)

        means = self.mean_params(z)
        logvars = self.logvar_params(z)
        std = torch.exp(0.5 * logvars)
        eps = torch.randn_like(logvars)
        if self.inference_mode:
            eps = torch.zeros_like(eps)
        z_sampled = means + eps * std

        return z_sampled, means, std

    def decode(self, z):
        """Do a forward pass of the VAE. Generates a reconstructed image based on z
        Parameters
        ----------
        z: torch.Tensor
            The latent vector.
        """
        img_recon = self.img_decoder(z)
        if self.with_logits:
            return torch.sigmoid(img_recon)
        return img_recon

    def set_inference_mode(self, mode):
        self.inference_mode = mode