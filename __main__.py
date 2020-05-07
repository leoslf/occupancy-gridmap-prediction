from gridmap.unet import *

if __name__ == "__main__":
    model = UNet()
    history = model.fit_df()
    
