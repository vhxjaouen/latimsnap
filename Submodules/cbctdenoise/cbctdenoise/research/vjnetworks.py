#  Vincent Jaouen' I2I networks with MONAI
#  The generator is MONAI's UNet
#  The discriminator is a multiscale conv encoder (à la pix2pixHD)
#  vincent.jaouen@imt-atlantique.fr

import torch 
import torch.nn as nn
from torch.nn import MSELoss, L1Loss
import torch.nn.functional as F

from monai.networks.blocks import ResBlock
from generative.losses import PatchAdversarialLoss
import monai.networks.nets as nets
adversarial_loss = PatchAdversarialLoss(criterion="bce") # bce, least_squares
gpu_device = torch.device(f'cuda:{0}')
from generative.networks.nets import MultiScalePatchDiscriminator
import kornia 
from kornia.filters import SpatialGradient
adversarial_loss = PatchAdversarialLoss(criterion="bce") # bce, least_squares
    
def identity_loss(original_images, translated_images):
    loss = F.l1_loss(translated_images, original_images)
    return loss# Adversarial loss for generators

class PatchGANDiscriminator(nn.Module):
    def __init__(self, in_channels, num_filters=64, num_layers=3, strides=None):
        super(PatchGANDiscriminator, self).__init__()
        print('(PatchGANDiscriminator) %d layers - %d filters' % (num_layers, num_filters))

        if strides is None:
            strides = [2] * (num_layers - 2) + [1, 1]  # Default to [2, 2, 2, 1, 1] if strides are not provided

        self.layers = nn.ModuleList()

        # Initial convolution layer
        self.layers.append(nn.Conv2d(in_channels, num_filters, kernel_size=4, stride=strides[0], padding=1))
        self.layers.append(nn.LeakyReLU(0.2, inplace=True))

        # Intermediate convolution layers
        for i in range(1, num_layers - 1):
            self.layers.append(nn.Conv2d(num_filters * 2**(i-1), num_filters * 2**i, kernel_size=4, stride=strides[i], padding=1))
            self.layers.append(nn.InstanceNorm2d(num_filters * 2**i))
            self.layers.append(nn.LeakyReLU(0.2, inplace=True))

        # Output layer
        self.layers.append(nn.Conv2d(num_filters * 2**(num_layers-2), 1, kernel_size=4, stride=strides[-1], padding=1))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
class MultiScaleDiscriminator(nn.Module):
    def __init__(self, in_channels, num_d=3, num_filters=64, num_layers_d=3):
        super(MultiScaleDiscriminator, self).__init__()
        print('(MultiScaleDiscriminator) %d discriminators - %d layers - %d filters' % (num_d, num_layers_d, num_filters))
        self.num_d = num_d
        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)
        self.discriminators = nn.ModuleList()
        
        # Create multiple discriminators
        for _ in range(num_d):
            self.discriminators.append(PatchGANDiscriminator(in_channels, num_filters, num_layers_d))
    
    def forward(self, x):
        outputs = []
        for discriminator in self.discriminators:
            outputs.append(discriminator(x))
            # Downsample for the next discriminator
            x = self.downsample(x)
        return outputs  # list of outputs from all discriminators
    
    

class MultiScaleDiscriminatorStrided(nn.Module):
    def __init__(self, in_channels, num_d=3, num_filters=64, num_layers_d=3, downsample_filters=64):
        super(MultiScaleDiscriminatorStrided, self).__init__()
        self.num_d = num_d
        # Initial discriminator layers
        self.discriminators = nn.ModuleList()
        
        for _ in range(num_d):
            self.discriminators.append(PatchGANDiscriminator(in_channels, num_filters, num_layers_d))
            # After the first discriminator, increase the in_channels for subsequent ones
            # to match the output channels of the downsampling layer
            in_channels = downsample_filters
        
        # Learnable downsampling
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, downsample_filters, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True)
        )
    
    def forward(self, x):
        outputs = []
        for i, discriminator in enumerate(self.discriminators):
            if i > 0:  # Apply downsampling for all but the first discriminator
                x = self.downsample(x)
            outputs.append(discriminator(x))
        return outputs  # List of outputs from all discriminators

    
class MONAI_UNet(nn.Module):
    def __init__(self, num_res_units=6, use_tanh=False):
        super(MONAI_UNet, self).__init__()
        self.use_tanh = use_tanh
        self.unet = nets.Unet(
            spatial_dims=2,
            in_channels=1,
            out_channels=1,
            channels=(64, 128, 256, 512, 512, 512, 512, 512),
            strides=(2, 2, 2, 2, 2, 2, 2),
            num_res_units=num_res_units,
        ).to(gpu_device)
        
    def forward(self, x):
        out = self.unet(x)
        if self.use_tanh:
            out = torch.tanh(out)
        return out

class MONAI_UNet512(nn.Module):
    def __init__(self, num_res_units=6):
        super(MONAI_UNet512, self).__init__()
        self.unet = nets.Unet(
            spatial_dims=2,
            in_channels=1,
            out_channels=1,
            channels=(64, 128, 256, 512, 768, 768, 1024, 1024, 1024),
            strides=(1, 2, 2, 2, 2, 2, 2, 1),
            num_res_units=num_res_units,
        ).to(gpu_device)
        
    def forward(self, x):
        return self.unet(x)

class MONAI_UNet128(nn.Module):
    def __init__(self, num_res_units=6):
        super(MONAI_UNet128, self).__init__()
        self.unet = nets.Unet(
            spatial_dims=2,
            in_channels=1,
            out_channels=1,
            channels=(64, 128, 256, 512, 512, 512, 512),
            strides=(2, 2, 2, 2, 2, 2),
            num_res_units=num_res_units,
        ).to(gpu_device)
        
    def forward(self, x):
        return self.unet(x)   



import torch
import torch.nn as nn
import kornia.losses

class L1_SSIM_Loss(nn.Module):
    def __init__(self, alpha=0.84):  # alpha ∈ [0,1]: weight for L1 vs SSIM
        super().__init__()
        self.l1 = nn.L1Loss()
        self.ssim = kornia.losses.SSIMLoss(window_size=11, reduction='mean')
        self.alpha = alpha

    def forward(self, x, y):
        # Rescale to [0,1] for SSIM
        x_ssim = (x + 1) / 2
        y_ssim = (y + 1) / 2
        
        l1_loss = self.l1(x, y)
        ssim_loss = self.ssim(x_ssim, y_ssim)
        
        # Combine them: weighted sum
        loss = self.alpha * l1_loss + (1 - self.alpha) * ssim_loss
        return loss


class Pix2Pix(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 num_d=1, 
                 num_filters_d=64, 
                 num_res_units_G=9,
                 num_layers_d=3,
                 use_tanh=False,
                 gan_mode='bce',
                ):
        super(Pix2Pix, self).__init__()
        self.generator_A_to_B = MONAI_UNet(num_res_units=num_res_units_G, use_tanh=use_tanh).to(gpu_device)
        self.discriminator_B = MultiScaleDiscriminator(in_channels=1, num_d=num_d, num_filters=num_filters_d, num_layers_d=num_layers_d ).to(gpu_device)
        self.criterionL1 = torch.nn.L1Loss()
        # 'least_squares' (LSGAN) is far more stable than 'bce' for large-domain-shift I2I (e.g. MR->CT)
        self.adv_loss_fn = PatchAdversarialLoss(criterion=('least_squares' if gan_mode == 'lsgan' else 'bce'))
        

    def calculate_edge_loss(self, img1, img2, alpha_NGF):
        from kornia.filters import SpatialGradient
        grad_src = SpatialGradient()(img1)
        grad_tgt = SpatialGradient()(img2)

        src_x = grad_src[:,:,0,:,:]
        src_y = grad_src[:,:,1,:,:]
        tgt_x = grad_tgt[:,:,0,:,:]
        tgt_y = grad_tgt[:,:,1,:,:]

        gradmag_src = torch.sqrt(torch.pow(src_x,2)+torch.pow(src_y,2)+alpha_NGF**2)
        gradmag_tgt = torch.sqrt(torch.pow(tgt_x,2)+torch.pow(tgt_y,2)+alpha_NGF**2)
        eps = 1e-8
        NGF = 1-1/2*(torch.pow((src_x/(gradmag_src+eps)*tgt_x/(gradmag_tgt+eps) + src_y/(gradmag_src+eps)*tgt_y/(gradmag_tgt+eps)),2))

        NGFM = torch.mean(NGF)

        return NGFM
    
    
    def compute_l1_loss(self, fake_B, real_B):
        l1_loss = self.criterionL1(real_B, fake_B)        
        return l1_loss  

    def compute_l1_ssim_loss(self, fake_B, real_B, alpha=0.84):
        l1_ssim_loss = L1_SSIM_Loss(alpha=alpha)
        loss = l1_ssim_loss(fake_B, real_B)
        return loss
    
    def compute_adv_loss(self, pred_fake_B):
        adv_loss = self.adv_loss_fn(pred_fake_B, target_is_real=True, for_discriminator=False)  
        return adv_loss  
        

    def compute_NGF_loss(self, fake_B, real_B, alpha_NGF):
        NGF_loss = self.calculate_edge_loss(real_B, fake_B, alpha_NGF)
        return NGF_loss         
    
    def compute_identity_loss(self, real_B):
        identity_B = self.generator_A_to_B(real_B)
        identity_loss =  self.criterionL1(real_B, identity_B)
        return identity_loss     
    
    def compute_discriminator_loss(self, real_B, fake_B):
        # Adversarial loss for discriminators
        pred_real_B = self.discriminator_B(real_B)
        pred_fake_B = self.discriminator_B(fake_B.detach())  # Detach fake_B from the computation graph

        discriminator_B_loss_real = self.adv_loss_fn(pred_real_B, target_is_real=True, for_discriminator=True)
        discriminator_B_loss_fake = self.adv_loss_fn(pred_fake_B, target_is_real=False, for_discriminator=True)

        total_discriminator_loss = (
            discriminator_B_loss_real
            + discriminator_B_loss_fake
        )
        return total_discriminator_loss

    def forward(self, real_A, real_B=None, is_training=True):
        # Translate images from domain A to domain B
        fake_B = self.generator_A_to_B(real_A)

        # # Identity mapping (optional)
        if is_training:
            identity_B = self.generator_A_to_B(real_B)

        # Adversarial outputs
        pred_fake_B = self.discriminator_B(fake_B)

        # return fake_B, fake_A, identity_A, identity_B, pred_fake_A, pred_fake_B 
        if is_training:
            return fake_B, identity_B, pred_fake_B    
        else:
            return fake_B, pred_fake_B

    
class CycleGAN(nn.Module):
    """CycleGAN sharing the SAME architecture as the NEC Pix2Pix model.

    Generators: MONAI_UNet(num_res_units=num_res_units_G) for both directions.
    Discriminators: MultiScaleDiscriminator(num_d, num_layers_d, num_filters_d)
    for both domains. This keeps the generator/discriminator identical to
    `Pix2Pix` so the only difference vs. NEC is the loss (cycle instead of NGF).
    """
    def __init__(self,
                 in_channels=1,
                 out_channels=1,
                 num_d=2,
                 num_layers_d=3,
                 num_filters_d=64,
                 num_res_units_G=9,
                 lambda_gan=1,
                 lambda_identity=5.0,
                 lambda_cyc=10.0,
                 use_tanh=False,
                 gan_mode='bce'):
        super(CycleGAN, self).__init__()
        self.lambda_gan = lambda_gan
        self.lambda_cyc = lambda_cyc
        self.lambda_identity = lambda_identity
        # Same generator architecture as Pix2Pix (NEC)
        self.G_AB = MONAI_UNet(num_res_units=num_res_units_G, use_tanh=use_tanh).to(gpu_device)
        self.G_BA = MONAI_UNet(num_res_units=num_res_units_G, use_tanh=use_tanh).to(gpu_device)
        # Same multi-scale discriminator architecture as Pix2Pix (NEC)
        self.D_B = MultiScaleDiscriminator(in_channels=1, num_d=num_d, num_filters=num_filters_d, num_layers_d=num_layers_d).to(gpu_device)
        self.D_A = MultiScaleDiscriminator(in_channels=1, num_d=num_d, num_filters=num_filters_d, num_layers_d=num_layers_d).to(gpu_device)
        self.criterionL1 = torch.nn.L1Loss()
        # 'least_squares' (LSGAN) is the standard, stable choice for CycleGAN
        self.adv_loss_fn = PatchAdversarialLoss(criterion=('least_squares' if gan_mode == 'lsgan' else 'bce'))

    def compute_generator_loss(self, fake_B, real_B, fake_A, real_A, cyc_B, cyc_A):
        # Adversarial loss for generators (fool each domain discriminator)
        pred_fake_B = self.D_B(fake_B)
        pred_fake_A = self.D_A(fake_A)

        adv_loss_B = self.adv_loss_fn(pred_fake_B, target_is_real=True, for_discriminator=False)
        adv_loss_A = self.adv_loss_fn(pred_fake_A, target_is_real=True, for_discriminator=False)

        cyc_loss_B = self.criterionL1(real_B, cyc_B)
        cyc_loss_A = self.criterionL1(real_A, cyc_A)

        # Identity loss (optional)
        identity_B = self.G_AB(real_B)
        identity_loss_B = self.criterionL1(real_B, identity_B)
        identity_A = self.G_BA(real_A)
        identity_loss_A = self.criterionL1(real_A, identity_A)

        return cyc_loss_B, identity_loss_B, adv_loss_B, cyc_loss_A, identity_loss_A, adv_loss_A

    def compute_discriminator_loss(self, real_B, fake_B, real_A, fake_A):
        # Adversarial loss for discriminators
        pred_real_B = self.D_B(real_B)
        pred_fake_B = self.D_B(fake_B.detach())  # Detach fake_B from the computation graph

        D_B_loss_real = self.adv_loss_fn(pred_real_B, target_is_real=True, for_discriminator=True)
        D_B_loss_fake = self.adv_loss_fn(pred_fake_B, target_is_real=False, for_discriminator=True)

        pred_real_A = self.D_A(real_A)
        pred_fake_A = self.D_A(fake_A.detach())  # Detach fake_A from the computation graph

        D_A_loss_real = self.adv_loss_fn(pred_real_A, target_is_real=True, for_discriminator=True)
        D_A_loss_fake = self.adv_loss_fn(pred_fake_A, target_is_real=False, for_discriminator=True)

        total_discriminator_loss = (
            D_B_loss_real
            + D_B_loss_fake
            + D_A_loss_real
            + D_A_loss_fake
        )
        return total_discriminator_loss

    def forward(self, real_A, real_B=None, is_training=True):
        # Translate images from domain A to domain B
        fake_B = self.G_AB(real_A)
        cyc_A = self.G_BA(fake_B)

        if is_training:
            fake_A = self.G_BA(real_B)
            cyc_B = self.G_AB(fake_A)
            # Identity mapping (optional)
            identity_B = self.G_AB(real_B)
            identity_A = self.G_BA(real_A)

            # Adversarial outputs
            pred_fake_B = self.D_B(fake_B)
            pred_fake_A = self.D_A(fake_A)

            return fake_B, identity_B, pred_fake_B, fake_A, identity_A, pred_fake_A, cyc_B, cyc_A
        else:
            # Inference: only the A -> B (e.g. MR -> CT) direction is needed
            pred_fake_B = self.D_B(fake_B)
            return fake_B, pred_fake_B
class Pix2PixNGF(Pix2Pix):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 num_d=1, 
                 num_filters_d=64, 
                 num_res_units_G=9,
                ):
        super(Pix2PixNGF, self).__init__(in_channels=in_channels, 
                                         out_channels=out_channels, 
                                         num_d=num_d, 
                                         num_filters_d=num_filters_d, 
                                         num_res_units_G=num_res_units_G)
        self.alpha_NGF = nn.Parameter(torch.tensor(0.1), requires_grad=True)

        
# SD for Strided Discriminator
class Pix2PixSD(Pix2Pix):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 num_d=1, 
                 num_filters_d=64, 
                 num_res_units_G=9,
                 num_layers_d=3,
                 # You might add additional parameters specific to the new discriminator here
                ):
        # Initialize the parent class with all the required arguments
        super(Pix2PixSD, self).__init__(in_channels=in_channels, 
                                              out_channels=out_channels, 
                                              num_d=num_d, 
                                              num_filters_d=num_filters_d, 
                                              num_res_units_G=num_res_units_G,
                                              num_layers_d=num_layers_d)
        
        # Here you redefine the discriminator_B with the new discriminator you wish to use
        self.discriminator_B = MultiScaleDiscriminatorStrided(in_channels=1, 
                                                num_d=num_d, 
                                                num_filters=num_filters_d, 
                                                num_layers_d=num_layers_d).to(gpu_device)
class Pix2Pix512(Pix2Pix):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 num_d=1, 
                 num_filters_d=64, 
                 num_res_units_G=9,  # Adjust this if needed for your MONAI_UNet512
                 num_layers_d=3,
                ):
        super(Pix2Pix512, self).__init__(in_channels=in_channels, 
                                         out_channels=out_channels, 
                                         num_d=num_d, 
                                         num_filters_d=num_filters_d, 
                                         num_res_units_G=num_res_units_G, 
                                         num_layers_d=num_layers_d)

        # Replace the generator with MONAI_UNet512
        self.generator_A_to_B = MONAI_UNet512(num_res_units=num_res_units_G).to(gpu_device)
        # Assuming discriminator and other components remain unchanged

class Pix2Pix128(Pix2Pix):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 num_d=1, 
                 num_filters_d=64, 
                 num_res_units_G=9,  # Adjust this if needed for your MONAI_UNet512
                 num_layers_d=3,
                ):
        super(Pix2Pix128, self).__init__(in_channels=in_channels, 
                                         out_channels=out_channels, 
                                         num_d=num_d, 
                                         num_filters_d=num_filters_d, 
                                         num_res_units_G=num_res_units_G, 
                                         num_layers_d=num_layers_d)

        # Replace the generator with MONAI_UNet512
        self.generator_A_to_B = MONAI_UNet128(num_res_units=num_res_units_G).to(gpu_device)
        # Assuming discriminator and other components remain unchanged



# Generateur RRDB (Residual-in-Residual Dense Block)

# DenseBlock
class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, bn_size=4):
        super(DenseBlock, self).__init__()
        # bn_size est utilisé pour le bottleneck dans les convolutions 1x1
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, bn_size * growth_rate, kernel_size=1, stride=1, padding=0, bias=False),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        return torch.cat([x, out], 1) # Concaténer l'entrée avec la sortie
    
# ResidualDenseBlock (RDB)
class ResidualDenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, num_dense_layers):
        super(ResidualDenseBlock, self).__init__()
        self.dense_layers = nn.ModuleList()
        current_channels = in_channels
        for i in range(num_dense_layers):
            self.dense_layers.append(DenseBlock(current_channels, growth_rate))
            current_channels += growth_rate # Les canaux augmentent à chaque dense block

        # Couche de convolution finale du bloc RDB
        self.conv1x1 = nn.Conv2d(current_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        identity = x
        for layer in self.dense_layers:
            x = layer(x)
        out = self.conv1x1(x)
        return out + identity # Connexion résiduelle

# Residual-in-Residual Dense Block (RRDB)
class RRDB(nn.Module):
    def __init__(self, in_channels, growth_rate, num_dense_layers, num_rdb):
        super(RRDB, self).__init__()
        self.rdb_layers = nn.ModuleList()
        for _ in range(num_rdb):
            self.rdb_layers.append(ResidualDenseBlock(in_channels, growth_rate, num_dense_layers))
        
        # Une convolution finale pour mixer les caractéristiques des RDB
        self.conv_final = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        identity = x
        for layer in self.rdb_layers:
            x = layer(x)
        # résidu à la sortie des RDB
        out = self.conv_final(x)
        return out * 0.2 + identity # Facteur d'échelle 0.2 comme dans ESRGAN
    
class SEBlock2D(nn.Module):
    """
    Squeeze-and-Excitation block for 2D inputs.
    """
    def __init__(self, channel, reduction=16):
        super(SEBlock2D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, max(1, channel // reduction), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(max(1, channel // reduction), channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class RRDBGenerator(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, num_rrdb=23, num_dense_layers=3, growth_rate=32, feature_channels=64, use_se=False):
        super(RRDBGenerator, self).__init__()

        self.use_se = use_se
        if self.use_se:
            self.se_input = SEBlock2D(in_channels)
            self.se_initial = SEBlock2D(feature_channels)
            self.se_trunk = SEBlock2D(feature_channels)

        self.conv_in = nn.Sequential(
            nn.Conv2d(in_channels, feature_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Empiler des RRDB
        rrdb_blocks = []
        for _ in range(num_rrdb):
            rrdb_blocks.append(RRDB(feature_channels, growth_rate, num_dense_layers, num_rdb=3)) # num_rdb=3 typique pour ESRGAN
        self.rrdb_trunk = nn.Sequential(*rrdb_blocks)

        self.conv_trunk = nn.Sequential(
            nn.Conv2d(feature_channels, feature_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.conv_out = nn.Conv2d(feature_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, x, is_training=True):
        if self.use_se:
            x = self.se_input(x)
        initial_features = self.conv_in(x)
        if self.use_se:
            initial_features = self.se_initial(initial_features)
        trunk_features = self.rrdb_trunk(initial_features)
        trunk_output = self.conv_trunk(trunk_features)
        if self.use_se:
            trunk_output = self.se_trunk(trunk_output)
        # Connexion résiduelle globale
        output = self.conv_out(initial_features + trunk_output) 
        return output
    
import kornia.losses

class Pix2PixRRDB(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 num_d=1, 
                 num_filters_d=64, 
                 num_res_units_G=9, 
                 num_layers_d=3,
                 # paramètres pour RRDBGenerator
                 num_rrdb_G=23, 
                 num_dense_layers_G=3, 
                 growth_rate_G=32, 
                 feature_channels_G=64,
                 use_se=False
                ):
        super(Pix2PixRRDB, self).__init__()
        self.generator_A_to_B = RRDBGenerator(
            in_channels=in_channels,
            out_channels=out_channels,
            num_rrdb=num_rrdb_G,
            num_dense_layers=num_dense_layers_G,
            growth_rate=growth_rate_G,
            feature_channels=feature_channels_G,
            use_se=use_se
        ).to(gpu_device)
        
        self.discriminator_B = MultiScaleDiscriminator(
            in_channels=out_channels, 
            num_d=num_d, 
            num_filters=num_filters_d, 
            num_layers_d=num_layers_d
        ).to(gpu_device)

        self.criterionL1 = torch.nn.L1Loss()
        self.criterionMS_SSIM_L1 = kornia.losses.MS_SSIMLoss().to(gpu_device)
        self.spatial_gradient = SpatialGradient().to(gpu_device)
        

    def calculate_edge_loss(self, img1, img2, alpha_NGF):
        grad_src = self.spatial_gradient(img1)
        grad_tgt = self.spatial_gradient(img2)

        src_x = grad_src[:,:,0,:,:]
        src_y = grad_src[:,:,1,:,:]
        tgt_x = grad_tgt[:,:,0,:,:]
        tgt_y = grad_tgt[:,:,1,:,:]

        gradmag_src = torch.sqrt(torch.pow(src_x,2)+torch.pow(src_y,2)+alpha_NGF**2)
        gradmag_tgt = torch.sqrt(torch.pow(tgt_x,2)+torch.pow(tgt_y,2)+alpha_NGF**2)
        eps = 1e-8
        NGF = 1-1/2*(torch.pow((src_x/(gradmag_src+eps)*tgt_x/(gradmag_tgt+eps) + src_y/(gradmag_src+eps)*tgt_y/(gradmag_tgt+eps)),2))

        NGFM = torch.mean(NGF)

        return NGFM
    
    def calculate_edge_loss_nonorm(self, img1, img2):
        grad_src = self.spatial_gradient(img1)
        grad_tgt = self.spatial_gradient(img2)

        src_x = grad_src[:,:,0,:,:]
        src_y = grad_src[:,:,1,:,:]
        tgt_x = grad_tgt[:,:,0,:,:]
        tgt_y = grad_tgt[:,:,1,:,:]

        GF = 1-1/2*(torch.pow((src_x*tgt_x + src_y*tgt_y),2))

        GFM = torch.mean(GF)

        return GFM
    
    def calculate_sobel_loss(self, img1, img2):
        grad_src = self.spatial_gradient(img1)
        grad_tgt = self.spatial_gradient(img2)

        src_x = grad_src[:,:,0,:,:]
        src_y = grad_src[:,:,1,:,:]
        tgt_x = grad_tgt[:,:,0,:,:]
        tgt_y = grad_tgt[:,:,1,:,:]

        sobel_loss = torch.mean(torch.abs(src_x - tgt_x) + torch.abs(src_y - tgt_y))
        
        return sobel_loss
    
    def compute_l1_ssim_loss(self, fake_B, real_B, alpha=0.84):
        l1_ssim_loss = L1_SSIM_Loss(alpha=alpha)
        loss = l1_ssim_loss(fake_B, real_B)
        return loss

    def compute_l1_mssim_loss(self, fake_B, real_B):
        loss = self.criterionMS_SSIM_L1(fake_B, real_B)
        return loss
        
    def compute_l1_loss(self, fake_B, real_B):
        l1_loss = self.criterionL1(real_B, fake_B)        
        return l1_loss 

    def compute_adv_loss(self, pred_fake_B):
        adv_loss = adversarial_loss(pred_fake_B, target_is_real=True, for_discriminator=False)  
        return adv_loss 
        
    def compute_sobel_loss(self, fake_B, real_B):
        sobel_loss = self.calculate_sobel_loss(real_B, fake_B)
        return sobel_loss
    
    def compute_NGF_loss(self, fake_B, real_B, alpha_NGF):
        NGF_loss = self.calculate_edge_loss(real_B, fake_B, alpha_NGF)
        return NGF_loss        
    
    def compute_GF_loss(self, fake_B, real_B):
        GF_loss = self.calculate_edge_loss_nonorm(real_B, fake_B)
        return GF_loss       
    
    def compute_identity_loss(self, real_B):
        identity_B = self.generator_A_to_B(real_B)
        identity_loss =  self.criterionL1(real_B, identity_B)
        return identity_loss    
    
    def compute_discriminator_loss(self, real_B, fake_B):
        pred_real_B = self.discriminator_B(real_B)
        pred_fake_B = self.discriminator_B(fake_B.detach())

        discriminator_B_loss_real = adversarial_loss(pred_real_B, target_is_real=True, for_discriminator=True)
        discriminator_B_loss_fake = adversarial_loss(pred_fake_B, target_is_real=False, for_discriminator=True)

        total_discriminator_loss = (
            discriminator_B_loss_real
            + discriminator_B_loss_fake
        )
        return total_discriminator_loss

    def forward(self, real_A, real_B=None, is_training=True):
        fake_B = self.generator_A_to_B(real_A)

        if is_training:
            if real_A.shape[1] == real_B.shape[1]:
                identity_B = self.generator_A_to_B(real_B)
            else:
                identity_B = None

        pred_fake_B = self.discriminator_B(fake_B)

        if is_training:
            return fake_B, identity_B, pred_fake_B    
        else:
            return fake_B, pred_fake_B




class FiLMBlock2D(nn.Module):
    """
    Spatially-Adaptive Normalization (SPADE-like) or FiLM using Modulating Prior (e.g. MRI).
    """
    def __init__(self, num_features):
        super(FiLMBlock2D, self).__init__()
        self.norm = nn.InstanceNorm2d(num_features, affine=False)
        
    def forward(self, x, gamma, beta):
        # x: CBCT features (B, C, H, W)
        # gamma, beta: scaling and shifting from MRI (B, C, H, W)
        normalized = self.norm(x)
        out = normalized * (1 + gamma) + beta
        return out

class MRIModulationNetwork(nn.Module):
    """
    Lightweight network to generate spatial gamma and beta maps from MRI.
    """
    def __init__(self, in_channels_mri=1, feature_channels=64):
        super(MRIModulationNetwork, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels_mri, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv_gamma = nn.Conv2d(32, feature_channels, kernel_size=3, stride=1, padding=1)
        self.conv_beta = nn.Conv2d(32, feature_channels, kernel_size=3, stride=1, padding=1)
        
        # Initialize gamma to 0, beta to 0, so initially it acts as identity
        nn.init.zeros_(self.conv_gamma.weight)
        nn.init.zeros_(self.conv_gamma.bias)
        nn.init.zeros_(self.conv_beta.weight)
        nn.init.zeros_(self.conv_beta.bias)

    def forward(self, x_mri):
        feat = self.conv1(x_mri)
        feat = self.conv2(feat)
        gamma = self.conv_gamma(feat)
        beta = self.conv_beta(feat)
        return gamma, beta

class RRDBGeneratorFiLM(nn.Module):
    def __init__(self, in_channels_cbct=3, in_channels_mri=1, out_channels=1, num_rrdb=23, num_dense_layers=3, growth_rate=32, feature_channels=64):
        super(RRDBGeneratorFiLM, self).__init__()
        
        self.conv_in = nn.Sequential(
            nn.Conv2d(in_channels_cbct, feature_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.2, inplace=True)
        )

        rrdb_blocks = []
        for _ in range(num_rrdb):
            rrdb_blocks.append(RRDB(feature_channels, growth_rate, num_dense_layers, num_rdb=3))
        self.rrdb_trunk = nn.Sequential(*rrdb_blocks)

        self.conv_trunk = nn.Sequential(
            nn.Conv2d(feature_channels, feature_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.mri_modulator = MRIModulationNetwork(in_channels_mri, feature_channels)
        self.film_block = FiLMBlock2D(feature_channels)

        self.conv_out = nn.Conv2d(feature_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, x_cbct, x_mri):
        initial_features = self.conv_in(x_cbct)
        
        if x_mri is not None:
            gamma, beta = self.mri_modulator(x_mri)
            modulated_initial = self.film_block(initial_features, gamma, beta)
            
            trunk_features = self.rrdb_trunk(modulated_initial)
            trunk_output = self.conv_trunk(trunk_features)
            
            output = self.conv_out(modulated_initial + trunk_output)
        else:
            trunk_features = self.rrdb_trunk(initial_features)
            trunk_output = self.conv_trunk(trunk_features)
            output = self.conv_out(initial_features + trunk_output)

        return output

class Pix2PixRRDBFiLM(Pix2PixRRDB):
    def __init__(self, 
                 in_channels_cbct=3, 
                 in_channels_mri=1,
                 out_channels=1, 
                 num_d=1, 
                 num_filters_d=64, 
                 num_res_units_G=9, 
                 num_layers_d=3,
                 num_rrdb_G=23, 
                 num_dense_layers_G=3, 
                 growth_rate_G=32, 
                 feature_channels_G=64
                ):
        super(Pix2PixRRDBFiLM, self).__init__(
            in_channels=in_channels_cbct+in_channels_mri,
            out_channels=out_channels,
            num_d=num_d,
            num_filters_d=num_filters_d,
            num_res_units_G=num_res_units_G,
            num_layers_d=num_layers_d,
            num_rrdb_G=num_rrdb_G,
            num_dense_layers_G=num_dense_layers_G,
            growth_rate_G=growth_rate_G,
            feature_channels_G=feature_channels_G,
            use_se=False
        )
        
        # Override the generator with FiLM variant
        self.generator_A_to_B = RRDBGeneratorFiLM(
            in_channels_cbct=in_channels_cbct,
            in_channels_mri=in_channels_mri,
            out_channels=out_channels,
            num_rrdb=num_rrdb_G,
            num_dense_layers=num_dense_layers_G,
            growth_rate=growth_rate_G,
            feature_channels=feature_channels_G
        ).to(gpu_device)
        
    def forward(self, x_cbct, x_mri, real_B=None, is_training=True):
        fake_B = self.generator_A_to_B(x_cbct, x_mri)
        
        if is_training:
            if real_B is not None and x_cbct.shape[1] == real_B.shape[1]:
                identity_B = self.generator_A_to_B(real_B, x_mri)
            else:
                identity_B = None

        pred_fake_B = self.discriminator_B(fake_B)

        if is_training:
            return fake_B, identity_B, pred_fake_B    
        else:
            return fake_B, pred_fake_B

class SpatialTemporalGate(nn.Module):
    def __init__(self, in_channels=3, hidden_channels=16):
        super(SpatialTemporalGate, self).__init__()
        self.attention_net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_channels, in_channels, kernel_size=3, padding=1, bias=True)
        )

    def forward(self, x):
        attn_logits = self.attention_net(x)
        spatial_weights = F.softmax(attn_logits, dim=1)
        weighted_cbct = x * spatial_weights
        return weighted_cbct

class RRDBGeneratorFiLM_Gate(nn.Module):
    def __init__(self, in_channels_cbct=3, in_channels_mri=1, out_channels=1, num_rrdb=23, num_dense_layers=3, growth_rate=32, feature_channels=64):
        super(RRDBGeneratorFiLM_Gate, self).__init__()
        
        self.spatial_gate = SpatialTemporalGate(in_channels_cbct, hidden_channels=16)
        
        self.conv_in = nn.Sequential(
            nn.Conv2d(in_channels_cbct, feature_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.mri_modulator = MRIModulationNetwork(in_channels_mri, feature_channels)
        self.film_block = FiLMBlock2D(feature_channels)

        rrdb_blocks = []
        for _ in range(num_rrdb):
            rrdb_blocks.append(RRDB(feature_channels, growth_rate, num_dense_layers, num_rdb=3))
        self.rrdb_trunk = nn.Sequential(*rrdb_blocks)

        self.conv_trunk = nn.Sequential(
            nn.Conv2d(feature_channels, feature_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.conv_out = nn.Conv2d(feature_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, x_cbct, x_mri):
        gated_cbct = self.spatial_gate(x_cbct)
        initial_features = self.conv_in(gated_cbct)
        
        if x_mri is not None:
            gamma, beta = self.mri_modulator(x_mri)
            modulated_initial = self.film_block(initial_features, gamma, beta)
            
            trunk_features = self.rrdb_trunk(modulated_initial)
            trunk_output = self.conv_trunk(trunk_features)
            output = self.conv_out(modulated_initial + trunk_output)
        else:
            trunk_features = self.rrdb_trunk(initial_features)
            trunk_output = self.conv_trunk(trunk_features)
            output = self.conv_out(initial_features + trunk_output)

        return output

class Pix2PixRRDBFiLM_Gate(Pix2PixRRDB):
    def __init__(self, 
                 in_channels_cbct=3, 
                 in_channels_mri=1,
                 out_channels=1, 
                 num_d=1, 
                 num_filters_d=64, 
                 num_res_units_G=9, 
                 num_layers_d=3,
                 num_rrdb_G=23, 
                 num_dense_layers_G=3, 
                 growth_rate_G=32, 
                 feature_channels_G=64
                ):
        super(Pix2PixRRDBFiLM_Gate, self).__init__(
            in_channels=in_channels_cbct+in_channels_mri,
            out_channels=out_channels,
            num_d=num_d,
            num_filters_d=num_filters_d,
            num_res_units_G=num_res_units_G,
            num_layers_d=num_layers_d,
            num_rrdb_G=num_rrdb_G,
            num_dense_layers_G=num_dense_layers_G,
            growth_rate_G=growth_rate_G,
            feature_channels_G=feature_channels_G,
            use_se=False
        )
        
        # Override the generator with FiLM + Gate variant
        self.generator_A_to_B = RRDBGeneratorFiLM_Gate(
            in_channels_cbct=in_channels_cbct,
            in_channels_mri=in_channels_mri,
            out_channels=out_channels,
            num_rrdb=num_rrdb_G,
            num_dense_layers=num_dense_layers_G,
            growth_rate=growth_rate_G,
            feature_channels=feature_channels_G
        ).to(gpu_device)
        
    def forward(self, x_cbct, x_mri, real_B=None, is_training=True):
        fake_B = self.generator_A_to_B(x_cbct, x_mri)
        
        if is_training:
            if real_B is not None and x_cbct.shape[1] == real_B.shape[1]:
                identity_B = self.generator_A_to_B(real_B, x_mri)
            else:
                identity_B = None

        pred_fake_B = self.discriminator_B(fake_B)

        if is_training:
            return fake_B, identity_B, pred_fake_B    
        else:
            return fake_B, pred_fake_B

class LSKA(nn.Module):
    """
    Large Separable Kernel Attention.
    Provides a massive receptive field (effective 23x23) to see whole streak artifacts
    while scaling linearly to protect VRAM.
    """
    def __init__(self, channels):
        super(LSKA, self).__init__()
        # Step 1: Local Context (Separated 5x5)
        self.conv0_h = nn.Conv2d(channels, channels, kernel_size=(1, 5), padding=(0, 2), groups=channels)
        self.conv0_v = nn.Conv2d(channels, channels, kernel_size=(5, 1), padding=(2, 0), groups=channels)
        
        # Step 2: Long-Range Context (Separated 7x7 with dilation 3 -> Effective 23x23)
        self.conv_spatial_h = nn.Conv2d(channels, channels, kernel_size=(1, 7), stride=(1, 1), padding=(0, 9), groups=channels, dilation=(1, 3))
        self.conv_spatial_v = nn.Conv2d(channels, channels, kernel_size=(7, 1), stride=(1, 1), padding=(9, 0), groups=channels, dilation=(3, 1))
        
        # Step 3: Channel Mixing
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x):
        attn = self.conv0_h(x)
        attn = self.conv0_v(attn)
        attn = self.conv_spatial_h(attn)
        attn = self.conv_spatial_v(attn)
        attn = self.conv1(attn)
        return attn

class LSKAGate(nn.Module):
    """
    Applies LSKA to the concatenated multi-modal input.
    """
    def __init__(self, in_channels):
        super(LSKAGate, self).__init__()
        self.lska = LSKA(in_channels)

    def forward(self, x):
        # x shape: (Batch, 4, Height, Width) [3 CBCT + 1 MRI]
        attn_logits = self.lska(x)
        # Sigmoid allows independent spatial gating for MRI and CBCTs
        spatial_weights = torch.sigmoid(attn_logits) 
        return x * spatial_weights

class RRDBGenerator_LSKA(nn.Module):
    def __init__(self, in_channels=4, out_channels=1, num_rrdb=23, num_dense_layers=3, growth_rate=32, feature_channels=64):
        super(RRDBGenerator_LSKA, self).__init__()
        
        # 1. The LSKA Spatial Gate
        self.lska_gate = LSKAGate(in_channels)
        
        # 2. Initial Feature Extraction
        self.conv_in = nn.Sequential(
            nn.Conv2d(in_channels, feature_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # 3. Deep Density Refinement Trunk
        rrdb_blocks = []
        for _ in range(num_rrdb):
            rrdb_blocks.append(RRDB(feature_channels, growth_rate, num_dense_layers, num_rdb=3))
        self.rrdb_trunk = nn.Sequential(*rrdb_blocks)

        self.conv_trunk = nn.Sequential(
            nn.Conv2d(feature_channels, feature_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.conv_out = nn.Conv2d(feature_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, x):
        # A. Dynamically weight localized spatial quality (streak dodging)
        gated_x = self.lska_gate(x)
        
        # B. Standard RRDB processing
        initial_features = self.conv_in(gated_x)
        trunk_features = self.rrdb_trunk(initial_features)
        trunk_output = self.conv_trunk(trunk_features)
        output = self.conv_out(initial_features + trunk_output)
        
        return output

class Pix2PixRRDB_LSKA(Pix2PixRRDB):
    def __init__(self, 
                 in_channels=4, 
                 out_channels=1, 
                 num_d=1, 
                 num_filters_d=64, 
                 num_res_units_G=9, 
                 num_layers_d=3,
                 num_rrdb_G=23, 
                 num_dense_layers_G=3, 
                 growth_rate_G=32, 
                 feature_channels_G=64
                ):
        # Initialize parent
        super(Pix2PixRRDB_LSKA, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            num_d=num_d,
            num_filters_d=num_filters_d,
            num_res_units_G=num_res_units_G,
            num_layers_d=num_layers_d,
            num_rrdb_G=num_rrdb_G,
            num_dense_layers_G=num_dense_layers_G,
            growth_rate_G=growth_rate_G,
            feature_channels_G=feature_channels_G,
            use_se=False # We handle attention explicitly now
        )
        
        # Override generator
        self.generator_A_to_B = RRDBGenerator_LSKA(
            in_channels=in_channels,
            out_channels=out_channels,
            num_rrdb=num_rrdb_G,
            num_dense_layers=num_dense_layers_G,
            growth_rate=growth_rate_G,
            feature_channels=feature_channels_G
        ).to(gpu_device)

class ResidualSE(nn.Module):
    """
    Channel-wise Squeeze-and-Excitation using a residual design:
    output_weights = 1 + scale * tanh(MLP(x))
    The last MLP layer is initialized to zero to act as identity.
    """
    def __init__(self, channels, reduction=16, se_scale=0.5):
        super(ResidualSE, self).__init__()
        self.se_scale = se_scale
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        hidden_dim = max(1, channels // reduction)
        
        self.fc1 = nn.Conv2d(channels, hidden_dim, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(hidden_dim, channels, kernel_size=1, bias=True)
        
        # Initialize zero to preserve identity mappings at start
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        stats = self.avg_pool(x)
        logits = self.fc2(self.relu(self.fc1(stats)))
        # Outputs weights around 1.0 (from 1 - scale to 1 + scale)
        weights = 1.0 + self.se_scale * torch.tanh(logits)
        return x * weights


class ReliabilityAwareFusionGate(nn.Module):
    """
    Combines both local/global spatial gating (via LSKA) and channel 
    recalibration (via ResidualSE). Modulates the features residually.
    """
    def __init__(self, channels, se_scale=0.5, spatial_scale=0.5):
        super(ReliabilityAwareFusionGate, self).__init__()
        self.spatial_scale = spatial_scale
        self.se = ResidualSE(channels, reduction=16, se_scale=se_scale)
        self.lska = LSKA(channels)
        
        self.spatial_proj = nn.Conv2d(channels, 1, kernel_size=1, bias=True)
        # Initialize spatial projection to 0 (acts as identity mapping initially)
        nn.init.zeros_(self.spatial_proj.weight)
        nn.init.zeros_(self.spatial_proj.bias)

    def forward(self, x):
        # Channel recalibration
        x_se = self.se(x)
        
        # Spatial recalibration
        spatial_features = self.lska(x_se)
        spatial_logits = self.spatial_proj(spatial_features) 
        spatial_weights = 1.0 + self.spatial_scale * torch.tanh(spatial_logits)
        
        return x_se * spatial_weights


class MultiModalRRDBGenerator_LSKA(nn.Module):
    def __init__(self, modalities, out_channels=1, num_rrdb=23, num_dense_layers=3, 
                 growth_rate=32, feature_channels=64, stem_feature_channels=16):
        super(MultiModalRRDBGenerator_LSKA, self).__init__()
        self.modalities = modalities
        num_modalities = len(modalities)
        
        # Create independent stems for each modality
        self.stems = nn.ModuleDict()
        for mod_name in modalities:
            # Note: Do not place `.to(gpu_device)` inside constructors 
            self.stems[mod_name] = nn.Sequential(
                nn.Conv2d(1, stem_feature_channels, kernel_size=3, padding=1, bias=True),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(stem_feature_channels, stem_feature_channels, kernel_size=3, padding=1, bias=True),
                nn.LeakyReLU(0.2, inplace=True)
            )
            
        fusion_channels = stem_feature_channels * num_modalities
        
        # Point-wise projection from fusion_channels down to feature_channels (independent of N modalities)
        self.fusion_proj = nn.Sequential(
            nn.Conv2d(fusion_channels, feature_channels, kernel_size=1, padding=0, bias=True),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # The new reliability-aware fusion gate
        self.reliability_gate = ReliabilityAwareFusionGate(feature_channels, se_scale=0.5, spatial_scale=0.5)

        # Legacy RRDB Trunk initialization
        rrdb_blocks = []
        for _ in range(num_rrdb):
            rrdb_blocks.append(RRDB(feature_channels, growth_rate, num_dense_layers, num_rdb=3))
        self.rrdb_trunk = nn.Sequential(*rrdb_blocks)

        self.conv_trunk = nn.Sequential(
            nn.Conv2d(feature_channels, feature_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.conv_out = nn.Conv2d(feature_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, x):
        # We assert the number of input channels matches the defined modalities at inference time
        B, C, H, W = x.shape
        assert C == len(self.modalities), f"Input channels ({C}) do not match the expected number of modalities ({len(self.modalities)})."
        
        # 1. Modality-specific stems
        stem_outputs = []
        for i, mod_name in enumerate(self.modalities):
            # Extract i-th modality and pass it through its relative stem backbone
            mod_input = x[:, i:i+1, :, :]
            stem_feat = self.stems[mod_name](mod_input)
            stem_outputs.append(stem_feat)
        
        # 2. Feature-level concatenate
        fused_features = torch.cat(stem_outputs, dim=1)
        
        # 3. 1x1 Feature Projection
        projected_features = self.fusion_proj(fused_features)
        
        # 4. Reliability Gate
        gated_features = self.reliability_gate(projected_features)
        
        # 5. Native RRDB Trunk Extractor (Residual connections logic)
        trunk_features = self.rrdb_trunk(gated_features)
        trunk_output = self.conv_trunk(trunk_features)
        output = self.conv_out(gated_features + trunk_output)
        
        return output

class Pix2PixRRDB_MultiModalLSKA(Pix2PixRRDB):
    def __init__(self, 
                 modalities,
                 out_channels=1, 
                 num_d=1, 
                 num_filters_d=64, 
                 num_res_units_G=9, 
                 num_layers_d=3,
                 num_rrdb_G=23, 
                 num_dense_layers_G=3, 
                 growth_rate_G=32, 
                 feature_channels_G=64,
                 stem_feature_channels_G=16
                ):
        """
        Pix2PixRRDB architecture natively modified to support Multi-Modal representations, utilizing 
        independent sequential modality extractors alongside a global reliability-aware fusion.
        """
        # We must align the expected input dimensions of the base discriminator 
        in_channels = len(modalities)

        super(Pix2PixRRDB_MultiModalLSKA, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            num_d=num_d,
            num_filters_d=num_filters_d,
            num_res_units_G=num_res_units_G,
            num_layers_d=num_layers_d,
            num_rrdb_G=num_rrdb_G,
            num_dense_layers_G=num_dense_layers_G,
            growth_rate_G=growth_rate_G,
            feature_channels_G=feature_channels_G,
            use_se=False # We handle our composite channel/spatial gates manually
        )
        
        # Override the Generator sequentially with our multimodal architecture
        self.generator_A_to_B = MultiModalRRDBGenerator_LSKA(
            modalities=modalities,
            out_channels=out_channels,
            num_rrdb=num_rrdb_G,
            num_dense_layers=num_dense_layers_G,
            growth_rate=growth_rate_G,
            feature_channels=feature_channels_G,
            stem_feature_channels=stem_feature_channels_G
        ) # the `.to(gpu_device)` should not be initialized inside, you must pipe this correctly at inst. )