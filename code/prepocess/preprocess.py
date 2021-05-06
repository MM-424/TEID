import yaml
import os
import numpy as np
import h5py
import math 
import torch
import torch.nn.functional as F

config = yaml.load(open('config.yml'),Loader=yaml.FullLoader)

# build path
base_path = 'D:/conv_motion_prediction/h36m/prepocess'
move_joint = np.array([0,1,2,3,4,5,6,7,8,9,10,12,13,14,15,17,18,19,21,22,25,26,27,29,30])
print(len(move_joint))
train_dataset = []
train_data_path = open(r'D:/motion_prediction_data/my_code/motion_prediction_code/gcn/net/utils/Data/train.txt')
train_save_path = os.path.join(base_path, 'train.npy')
train_save_path = train_save_path.replace("\\","/")


def velocity_vector(previous_one_frame, current_frame):

    vectors = torch.FloatTensor([])


    A = current_frame- previous_one_frame
    one_joint_vector = torch.tensor(A,dtype = torch.float32)
    vectors = torch.cat((vectors, one_joint_vector), dim=0)
    vectors = vectors.view(1, previous_one_frame.shape[0])
    return vectors

def loc_exchange(input):
    '''b:betach size; f:frames; n:nodes_num*3'''
    fr = input.shape[0]        #frame_number

    input = input.reshape(fr, -1, 3)
    nd = input.shape[1]        #node_num
    input = torch.tensor(input, dtype = torch.float32)
    total_vector = torch.FloatTensor([])
    one_sequence = torch.FloatTensor([])
    for a in range (input.shape[0]-1):
        one_frame = torch.FloatTensor([])
        for b in range (input.shape[1]):
            vector = velocity_vector(input[a,b],input[a+1,b])
            vector = torch.where(torch.isnan(vector), torch.full_like(vector, 0), vector)
            one_frame = torch.cat([one_frame,vector],dim=1)
        one_sequence = torch.cat([one_sequence,one_frame],dim=0)
    total_vector = torch.cat([total_vector,one_sequence],dim=0)
    
    total_vector = total_vector.view(fr-1, nd, 3)

    return total_vector


for train_one_data_path in train_data_path:
    
    keyword = 'Walking'
    if keyword in train_one_data_path:
        print (train_one_data_path)
        train_one_data_path = train_one_data_path.strip('\n')
        # load train data
        train_data = h5py.File(train_one_data_path,'r')
        coordinate_normalize_joint = train_data['coordinate_normalize_joint'][:,move_joint,:] 
        train_num = int(coordinate_normalize_joint.shape[0])
        coordinate_normalize_joint = torch.tensor(coordinate_normalize_joint)
        vector_velocity = loc_exchange(coordinate_normalize_joint)
        position_set = coordinate_normalize_joint[1:]
        position_set = torch.tensor(position_set, dtype = torch.float32)
        vector_velocity = torch.tensor(vector_velocity, dtype = torch.float32)

        velocity_position = torch.cat([vector_velocity, position_set],2)

        train_one_dataset = []
        for i in range(velocity_position.shape[0]):            
            train_data = velocity_position[i]
            train_data = np.array(train_data)
            train_one_dataset.append(train_data)
        train_one_dataset = np.array(train_one_dataset)
        print ('train_one_dataset:\n',train_one_dataset.shape)
        train_dataset.append(train_one_dataset)
    else:
        continue

train_dataset = np.array(train_dataset)
# save data
np.save(train_save_path, train_dataset)


test_data_path = open(r'D:/motion_prediction_data/my_code/motion_prediction_code/gcn/net/utils/Data/test.txt').readline()
print ('test_data_path:\n',test_data_path)
test_data_path = test_data_path[:-1]
test_save_path = os.path.join( base_path, 'test.npy')
test_save_path = test_save_path.replace("\\","/")

# load test data
test_data = h5py.File(test_data_path,'r')
test_coordinate_normalize_joint = test_data['coordinate_normalize_joint'][:,move_joint,:]
test_num = int(test_coordinate_normalize_joint.shape[0])
test_coordinate_normalize_joint = torch.tensor(test_coordinate_normalize_joint)

test_angle_velocity = loc_exchange(test_coordinate_normalize_joint)
test_vector_velocity = loc_exchange(test_coordinate_normalize_joint)

test_position_set = test_coordinate_normalize_joint[1:]
test_position_set = torch.tensor(test_position_set, dtype = torch.float32)
test_vector_velocity = torch.tensor(test_vector_velocity, dtype = torch.float32)
test_velocity_position = torch.cat([test_vector_velocity, test_position_set],2)

print ('test_dataset:\n',test_velocity_position.shape)

# save data
np.save(test_save_path, test_velocity_position)