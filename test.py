import torch
from config import config
from models.u_net import UNet
from torch import nn, optim
import utils
from torchvision import transforms
import cv2
import numpy as np
from PIL import Image


data_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.Grayscale(1),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,), std=(0.5,))
    ])

# dataset_frontiers = FrontierPredictionDataset(txt_file=config.txt_file,
#                                                 root_dir=config.root_dir,
#                                                 transform=data_transform)

# train_data, val_data, test_data = utils.dataset_split(dataset_frontiers, 0.8)

# train_loader = torch.utils.data.DataLoader(dataset=train_data,
#                                             batch_size=config.batch_size,
#                                             num_workers=config.num_workers,
#                                             shuffle=True)

# test_loader = torch.utils.data.DataLoader(dataset=test_data,
#                                             batch_size=config.batch_size,
#                                             num_workers=config.num_workers,
#                                             shuffle=True)


#net = UNet()
#net.cuda()
net = torch.load('/home/micrl/Documents/kay/catkin_ws/src/vae_map_prediction/src/predictor/250_model.pth')
net = nn.DataParallel(net)

# checkpoint = torch.load('./save/checkpoint.pth.tar')
# recorder = checkpoint['recorder']
# config.start_epoch = checkpoint['epoch']
# net.load_state_dict(checkpoint['state_dict'])
# optimizer.load_state_dict(checkpoint['optimizer'])
# utils.print_log("=> loaded checkpoint '{}' (epoch {})" .format(
#     config.resume, checkpoint['epoch']), log)

net.eval()

path = '/home/micrl/Documents/kay/catkin_ws/src/vae_map_prediction/dataset_2'
filename = '626_0_1.png'
img_path = path + '/submaps/' + filename
img_gt_path = path + '/submaps_gt/' + filename


img = Image.open(img_path)
#img = cv2.imread(img_path)
img_t = data_transform(img)
img_t = img_t.unsqueeze(0)
img_out = net(img_t)
output = img_out.cpu().squeeze(0)
img_output = transforms.ToPILImage()(output)
img_output.show()
img.show()


