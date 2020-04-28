# -*- coding: UTF-8 -*-
from models.residual_fully_conv_vae import ResidualFullyConvVAEModel

if __name__ == "__main__":
    model = ResidualFullyConvVAEModel()
    model.main_loop()
