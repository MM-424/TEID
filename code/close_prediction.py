import yaml
import h5py
import os
import time
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
import numpy as np
from torch.utils.data.dataloader import DataLoader
import velocity_model
from net.model import *

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')
config = yaml.load(open('./config.yml'),Loader=yaml.FullLoader)


node_num = config['node_num']
input_n=config['input_n']
output_n=config['output_n']
base_path = './prepocess'
input_size = config['in_features']
hidden_size = config['hidden_size']
output_size = config['out_features']
batch_size = config['batch_size']

test_save_path = os.path.join(base_path, 'test.npy')
test_save_path = test_save_path.replace("\\","/")
dataset = np.load(test_save_path,allow_pickle = True)
dataset = torch.tensor(dataset, dtype = torch.float32,requires_grad=False)

#load data
dataset = dataset[304:,:,:]


input_dataset = dataset[0:input_n]
output_dataset = dataset[input_n:input_n+output_n]
input_dataset = input_dataset.expand(batch_size,input_dataset.shape[0],input_dataset.shape[1],input_dataset.shape[2])
output_dataset = output_dataset.expand(batch_size,output_dataset.shape[0],output_dataset.shape[1],output_dataset.shape[2])

input_dataset = input_dataset
output_dataset = output_dataset


total_samples = 0
total_mse = 0
total_mpjpe = 0

model_x = torch.load(os.path.join(base_path, 'generator_x_4GRU.pkl')).to(device)
model_y = torch.load(os.path.join(base_path, 'generator_y_4GRU.pkl')).to(device)
model_z = torch.load(os.path.join(base_path, 'generator_z_4GRU.pkl')).to(device)


input_angle = input_dataset[:, :, :, :3]
target_angle = output_dataset[:, :, :, :3]

input_vector = input_dataset[:, :, :, :3]
target_vector = output_dataset[:, :, :, :3]

#read angle_x
input_x = input_velocity[:,:,:,0].permute(0, 2, 1).float()
target_x = target_velocity[:,:,:,0].float()

#read angle_y
input_y = input_velocity[:,:,:,1].permute(0, 2, 1).float()
target_y = target_velocity[:,:,:,1].float()

#read angle_z
input_z = input_velocity[:,:,:,2].permute(0, 2, 1).float()
target_z = target_velocity[:,:,:,2].float()

#read 3D data
input_3d_data = in_shots[:, :, :, 3:]
target_3d_data =out_shot[:, :, :, 3:]

output_x, _ = model_x(input_x, hidden_size)
output_x = output_x.view(target_x.shape[0],target_x.shape[2],output_size)

output_y, _ = model_y(input_y, hidden_size)
output_y = output_y.view(target_y.shape[0],target_y.shape[2],output_size)

output_z, _ = model_z(input_z, hidden_size)
output_z = output_z.view(target_z.shape[0],target_z.shape[2],output_size)

x = output_x.permute(0, 2, 1)
y = output_y.permute(0, 2, 1)
z = output_z.permute(0, 2, 1)

pred_set = torch.stack((x,y,z),3)

pred_set = pred_set.reshape(pred_set.shape[0],pred_set.shape[1],-1,3)

#reconstruction_loss
input_pose = torch.zeros((target_velocity.shape[0], output_n, input_3d_data.shape[-2], input_3d_data.shape[-1]))
for a in range(input_pose.shape[0]):
    input_pose[a,0,:,:] = input_3d_data[a,input_n-1,:,:]
re_data = torch.FloatTensor([])
for b in range (target_3d_data.shape[0]):
    for c in range (target_3d_data.shape[1]):
        reconstruction_coordinate = velocity_model.reconstruction_motion(pred_v[b,c,:,], pred_angle_set[b, c,:,:], input_pose[b, c, :, :],node_num)
        re_data = torch.cat([re_data,reconstruction_coordinate],dim=0)
        reconstruction_coordinate = reconstruction_coordinate
        if c+1<target_3d_data.shape[1]:
            input_pose[b,c+1,:,:] = reconstruction_coordinate
        else:
            continue
re_data = re_data.view(target_3d_data.shape[0],-1,node_num,3)

frame_re_data = re_data[0]
frame_target_3d_data = target_3d_data[0]

mpjpe_set = []
for i in range (frame_re_data.shape[0]):
    frame_re_data = frame_re_data.to(device)
    frame_target_3d_data = frame_target_3d_data.to(device)
    frame_rec_loss = torch.mean(torch.norm(frame_re_data[i] - frame_target_3d_data[i], 2, 1))
    mpjpe_set.append(frame_rec_loss)




#save vis data
frame_target_3d_data = frame_target_3d_data.cpu()
frame_re_data = frame_re_data.cpu()
frame_target_3d_data = np.array(frame_target_3d_data[0])
mpjpe_set = np.array(mpjpe_set)
vis_save_path = os.path.join(base_path, 'vis.npy')
vis_mpjpe_save_path = os.path.join(base_path, 'vis_mpjpe.npy')
np.save(vis_save_path, frame_re_data)
np.save(vis_mpjpe_save_path, mpjpe_set)
print ('-------------------') 
print ('mpjpe_set',mpjpe_set)   
print ('frame_re_data.shape:\n',frame_re_data.shape)



































