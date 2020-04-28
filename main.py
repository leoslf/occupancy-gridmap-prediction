# -*- coding: UTF-8 -*-
from models.u_net import UNetModel

if __name__ == "__main__":
    model = UNetModel()
    model.main_loop()
