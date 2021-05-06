import yaml
import h5py
import os
import torch
import torch.nn as nn
import numpy as np
#from progress.bar import Bar
from torch import optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from net.model import *
from prepocess.data_utils import *

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config = yaml.load(open('./config.yml'),Loader=yaml.FullLoader)

node_num = config['node_num']
input_n=config['input_n']
output_n=config['output_n']
base_path = './prepocess'
input_size = config['in_features']
hidden_size = config['hidden_size']
output_size = config['out_features']
lr=config['learning_rate']
batch_size = config['batch_size']


train_save_path = os.path.join(base_path, 'train.npy')
train_save_path = train_save_path.replace("\\","/")
dataset = np.load(train_save_path,allow_pickle = True)

#25node
chain = [[1], [132.95, 442.89, 454.21, 162.77, 75], [132.95, 442.89, 454.21, 162.77, 75],
         [233.58, 257.08, 121.13, 115], [257.08, 151.03, 278.88, 251.73, 100 ],
         [257.08,151.03, 278.88, 251.73, 100]]
for x in chain:
    s = sum(x)
    if s == 0:
        continue
    for i in range(len(x)):
        x[i] = (i+1)*sum(x[i:])/s

chain = [item for sublist in chain for item in sublist]
nodes_weight = torch.tensor(chain)
nodes_weight = nodes_weight.unsqueeze(1)
nodes_frame_weight = nodes_weight.expand(25, 25)

# frame_weight = torch.tensor([3, 2, 1.5, 1.5, 1, 0.5, 0.2, 0.2, 0.1, 0.1,
                             # 0.06, 0.06, 0.05, 0.05, 0.04, 0.04, 0.03, 0.03, 0.03, 0.02, 0.02,
                             # 0.02, 0.02, 0.02, 0.02])

frame_weight = torch.tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 5.00])

frame_weight = frame_weight.cuda()
nodes_frame_weight = nodes_frame_weight.cuda()

for epoch in range(config['train_epoches']):

    for i in range (dataset.shape[0]):
        data = dataset[i]

        train_data = LPDataset(data, input_n, output_n)

        train_loader = DataLoader(
            dataset=train_data,
            batch_size=config['batch_size'],
            shuffle=True,
            pin_memory=True,
            drop_last=True
        )
        model_x = Generator(input_size, hidden_size, output_size, batch_size)
        model_y = Generator(input_size, hidden_size, output_size, batch_size)
        model_z = Generator(input_size, hidden_size, output_size, batch_size)
        
        model_x.cuda()
        model_y.cuda()
        model_z.cuda()

        mse = nn.MSELoss(reduction='mean')
        print(">>> total params: {:.2f}M".format(sum(p.numel() for p in model_x.parameters()) / 1000000.0))

        optimizer_x = optim.Adam(model_x.parameters(), lr)
        optimizer_y = optim.Adam(model_y.parameters(), lr)
        optimizer_z = optim.Adam(model_z.parameters(), lr)

        print('pretrain generator')
        if os.path.exists(os.path.join(base_path, 'generator_x_4GRU.pkl')):
            print('---------------------------------')
            model_x.load_state_dict(torch.load(os.path.join(base_path, 'generator_x_4GRU.pkl')),strict=False)
            model_y.load_state_dict(torch.load(os.path.join(base_path, 'generator_y_4GRU.pkl')),strict=False)
            model_z.load_state_dict(torch.load(os.path.join(base_path, 'generator_z_4GRU.pkl')),strict=False)
            for i, data in enumerate(train_loader):
                optimizer_x.zero_grad()
                optimizer_y.zero_grad()
                optimizer_z.zero_grad()

                in_shots, out_shot = data
                
                in_shots = in_shots.cuda()
                out_shot = out_shot.cuda()
                
                input_velocity = in_shots[:, :, :, :3]
                target_velocity = out_shot[:, :, :, :3]


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

                loss_x = 0
                loss_y = 0
                loss_z = 0
                loss_rec = 0

                output_x, _ = model_x(input_x, hidden_size)
                output_x = output_x.view(target_x.shape[0],target_x.shape[2],output_size)
                target_x_loss = target_x.permute(0, 2, 1)
                loss_x += torch.mean(torch.norm((output_x- target_x_loss)*frame_weight*nodes_frame_weight, 2, 1))

                output_y, _ = model_y(input_y, hidden_size)
                output_y = output_y.view(target_y.shape[0],target_y.shape[2],output_size)
                target_y_loss = target_y.permute(0, 2, 1)
                loss_y += torch.mean(torch.norm((output_y- target_y_loss)*frame_weight*nodes_frame_weight, 2, 1))

                output_z, _ = model_z(input_z, hidden_size)
                output_z = output_z.view(target_z.shape[0],target_z.shape[2],output_size)
                target_z_loss = target_z.permute(0, 2, 1)
                loss_z += torch.mean(torch.norm((output_z- target_z_loss)*frame_weight*nodes_frame_weight, 2, 1))

                x = output_x.permute(0, 2, 1)
                y = output_y.permute(0, 2, 1)
                z = output_z.permute(0, 2, 1)
                
                pred_set = torch.stack((x,y,z),3)

                pred_set = pred_set.reshape(pred_set.shape[0],pred_set.shape[1],-1,3)

                total_loss = loss_x + loss_y + loss_z
                total_loss.backward()
                nn.utils.clip_grad_norm_(model_x.parameters(), config['gradient_clip'])
                nn.utils.clip_grad_norm_(model_y.parameters(), config['gradient_clip'])
                nn.utils.clip_grad_norm_(model_z.parameters(), config['gradient_clip'])

                optimizer_x.step()
                optimizer_y.step()
                optimizer_z.step()
                print('[epoch %d] [step %d] [loss %.4f]' % (epoch, i, total_loss.item()))
            torch.save(model_x.state_dict(), os.path.join(base_path, 'generator_x_4GRU.pkl'))
            torch.save(model_y.state_dict(), os.path.join(base_path, 'generator_y_4GRU.pkl'))
            torch.save(model_z.state_dict(), os.path.join(base_path, 'generator_z_4GRU.pkl'))


        else:
            for i, data in enumerate(train_loader):
                optimizer_x.zero_grad()
                optimizer_y.zero_grad()
                optimizer_z.zero_grad()

                in_shots, out_shot = data
                
                in_shots = in_shots.cuda()
                out_shot = out_shot.cuda()
                
                input_velocity = in_shots[:, :, :, :3]
                target_velocity = out_shot[:, :, :, :3]

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

                loss_x = 0
                loss_y = 0
                loss_z = 0
                loss_rec = 0
               
                output_x, _ = model_x(input_x, hidden_size)
                output_x = output_x.view(target_x.shape[0],target_x.shape[2],output_size)
                target_x_loss = target_x.permute(0, 2, 1)
                loss_x += torch.mean(torch.norm((output_x- target_x_loss)*frame_weight*nodes_frame_weight, 2, 1))

                output_y, _ = model_y(input_y, hidden_size)
                output_y = output_y.view(target_y.shape[0],target_y.shape[2],output_size)
                target_y_loss = target_y.permute(0, 2, 1)
                loss_y += torch.mean(torch.norm((output_y- target_y_loss)*frame_weight*nodes_frame_weight, 2, 1))

                output_z, _ = model_z(input_z, hidden_size)
                output_z = output_z.view(target_z.shape[0],target_z.shape[2],output_size)
                target_z_loss = target_z.permute(0, 2, 1)
                loss_z += torch.mean(torch.norm((output_z- target_z_loss)*frame_weight*nodes_frame_weight, 2, 1))

                x = output_x.permute(0, 2, 1)
                y = output_y.permute(0, 2, 1)
                z = output_z.permute(0, 2, 1)

                pred_set = torch.stack((x,y,z),3)
                pred_set = pred_set.reshape(pred_set.shape[0],pred_set.shape[1],-1,3)

                total_loss = loss_x + loss_y + loss_z

                total_loss.backward()
                nn.utils.clip_grad_norm_(model_x.parameters(), config['gradient_clip'])
                nn.utils.clip_grad_norm_(model_y.parameters(), config['gradient_clip'])
                nn.utils.clip_grad_norm_(model_z.parameters(), config['gradient_clip'])

                optimizer_x.step()
                optimizer_y.step()
                optimizer_z.step()
                print('[epoch %d] [step %d] [loss %.4f]' % (epoch, i, total_loss.item()))
            torch.save(model_x.state_dict(), os.path.join(base_path, 'generator_x_4GRU.pkl'))
            torch.save(model_y.state_dict(), os.path.join(base_path, 'generator_y_4GRU.pkl'))
            torch.save(model_z.state_dict(), os.path.join(base_path, 'generator_z_4GRU.pkl'))

torch.save(model_x, os.path.join(base_path, 'generator_x_4GRU.pkl'))
torch.save(model_y, os.path.join(base_path, 'generator_y_4GRU.pkl'))
torch.save(model_z, os.path.join(base_path, 'generator_z_4GRU.pkl'))
print ('Parameters are stored in the generator.pkl file')











































