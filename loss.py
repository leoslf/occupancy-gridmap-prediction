import ssim
import config
import torch


def loss_function(recon_x, x, loss_1, loss_2=None, mu=None, logvar=None):
    loss_types = ['MSE', 'BCE', 'SSIM']
    if loss_1 == loss_2:
        raise ValueError('loss_1 shound NOT equal to loss_2')
    if loss_1 not in loss_types:
        raise ValueError('loss_1 shound be MSE,BCE or SSIM')
    if loss_2 is not None and loss_2 not in loss_types:
        raise ValueError('loss_2 shound be MSE,BCE or SSIM')

    mse_loss = torch.nn.MSELoss(reduce = True, reduction='sum')
    bce_loss = torch.nn.BCELoss(reduce = True, reduction='sum')

    if loss_1 == 'BCE':
        loss = bce_loss(recon_x, x)
    elif loss_1 == 'MSE':
        loss = mse_loss(recon_x, x)
    elif loss_1 == 'SSIM':
        loss = -ssim.ssim(recon_x, x, nonnegative_ssim=True)

        
    if loss_2 is not None:
        if loss_2 == 'BCE':
            loss = 0.5*loss + 0.5*bce_loss(recon_x, x)
        elif loss_2 == 'MSE':
            loss = 0.5*loss + 0.5*mse_loss(recon_x, x)
        elif loss_2 == 'SSIM':
            loss = 0.8*loss - 0.2*ssim.ssim(recon_x, x, nonnegative_ssim=True)

    
    if mu is not None and logvar is not None:
        # KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
        # KLD = torch.sum(KLD_element).mul_(-0.5)
        KLD = -0.5 * torch.sum(1+logvar-torch.exp(logvar)-mu**2)
        loss = loss + KLD
        
    return loss