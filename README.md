# 2D Grid Map Prediction

To predict 2D occupancy grid map using UNet, VAE, and GAN.

## Dependencies:
* tensorflow keras

## Dataset

Link:
https://drive.google.com/file/d/1CbreYUMEB_5qMCehGTRd40O-ijO5S6rA/view?usp=sharing

It contains about 10,000 images.


## Usage
1. extract the dataset **frontiers_dataset.tar.gz** to any folder

2. modify the dataset path in config.yml

4. visualization via tensorboard
    ```
    tensorboard --logdir=runs
    ```
    then open the browser...

5. To execute the finalized version implemented in Keras

    ```
    python3 .
    ```

