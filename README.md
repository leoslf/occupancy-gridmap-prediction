# 2D Grid Map Prediction

To predict 2D occupancy grid map using UNet, VAE, and GAN.

## Dependencies:
* tensorflow keras
* pytorch
* tensorboardx

### Note :
I use python 2.7 for the convenience of working with ROS. It may work with python 3.x.
## Dataset

Link:
https://drive.google.com/file/d/1CbreYUMEB_5qMCehGTRd40O-ijO5S6rA/view?usp=sharing

It contains about 10,000 images.


## Usage
1. extract the dataset **frontiers_dataset.tar.gz** to any folder

2. modify the dataset path in config.yml

3. for Unet:
    ```
    python main.py
    ```
    for VAE:
    ```
    python main_vae.py
    ```
4. visualization via tensorboard
    ```
    tensorboard --logdir=runs
    ```
    then open the browser...

5. To execute the finalized version implemented in Keras

    ```
    python3 .
    ```

### Further notes ###

#### Code
* This dirty code needs refactoring...

* main.py and main_vae.py are almost the same. The loss calculation with latent variables is a bit different.

#### TODO
* accuracy has not been undefine yet, and I just use acc = 1-loss

* Network Tuning

* GAN

* use weighted loss (mask)
