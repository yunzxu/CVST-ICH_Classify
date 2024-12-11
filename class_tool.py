import torch
import numpy as np
from batch_aug import transforms_create
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def inverse_sigmoid(x):
    return 1 - 1 / (1 + torch.exp(-x))

def class_inference(test_case,model=None,is_patch=False,addmask=False,mask=None,augpred=False,multi_class=True,patch_size =[96,96,96],inverse_sig=False):
    '''
    inverse_sigmoid:False，就是正常，True就是变成[1,0]
    '''
    model.eval()
    # model1.eval()
    with torch.no_grad():

        
        #print(class_matrix.shape)
        if is_patch:
            x = test_case[np.newaxis,np.newaxis,:,:,:]
            
            x = torch.Tensor(x).to(device)
            if addmask:
                mask=np.array(mask[np.newaxis,np.newaxis,:,:,:],'int16')
                mask = torch.Tensor(mask).to(device)
                x =torch.cat([x,mask],dim=1)
            # print('x',x.shape)
            y = model(x)
            # y1 = model1(x)
            # y = (y+y1)/2
            if multi_class:
                y =  torch.softmax(y, dim=1)
            else:
                if  inverse_sig:
                    y = inverse_sigmoid(y)
                else:
                    y = torch.sigmoid(y)
            y = y.squeeze().detach().cpu().numpy()
            class_matrix =y
            if augpred:
                
                pred_list = []
                pred_list.append(class_matrix)
                for i in range(3):
                    data={'data':test_case[np.newaxis,np.newaxis,:,:,:],'seg':np.array(mask.cpu())}
                    data=transforms_create(data,patch_size=[128,128,128],batch_size=1,train_patch=True,pro=0.1)
                    X_patch=torch.Tensor(data['data']).to(device)
                    Y_patch=torch.Tensor(data['seg']).to(device)
                    #print('Y_patch',Y_patch.shape)
                    input =torch.cat([X_patch,Y_patch],dim=1)
                    y = model(input)
                    if multi_class:
                        y =  torch.softmax(y, dim=1)
                    else:
                        if  inverse_sig:
                            y = inverse_sigmoid(y)
                        else:
                            y = torch.sigmoid(y)
                    y = y.squeeze().detach().cpu().numpy()
                    pred_list.append(y)
                    class_matrix =np.array(pred_list)




        else:
            class_matrix = slide_windows_3D_classify(test_case,model,window_size=patch_size,step_size=[48,48,48],num_labels=3,device=torch.device("cuda"),addmask=addmask,mask=mask)



        # class_matrix[class_matrix>0.5]=1
        # class_matrix[class_matrix<0.5]=0
    
    return class_matrix

def slide_windows_3D_classify(input_volume,model,window_size=[96,96,96],step_size=[48,48,48],num_labels=3,device=torch.device("cuda"),addmask=False,mask=None):
    '''
    3D分类任务的滑动窗口
    '''
    # 读取输入3D图像
    # 定义滑动窗口的大小和步长
    # window_size = (64, 64, 64)
    # step_size = (32, 32, 32)

    # 计算输出3D图像的大小
    if mask.all() ==  None:
        mask = np.zeros([input_shape[0],input_shape[1],input_shape[2]])
    else:
        mask  =np.array(mask,'int16')
    input_shape = input_volume.shape
    output_index = [(input_shape[0]-window_size[0])//step_size[0]+1,
                    (input_shape[1]-window_size[1])//step_size[1]+1,
                    (input_shape[2]-window_size[2])//step_size[2]+1]
    output_shape =[output_index[0]*step_size[0]+window_size[0],output_index[1]*step_size[1]+window_size[1],output_index[2]*step_size[2]+window_size[2]]

   # print('slide_windows_3D shape',input_shape,output_shape)

    # 初始化输出3D图像
    output_volume = np.zeros(output_shape)
    inferce_map = np.zeros(output_shape) ##用于计算权重

    input_volume = np.pad(input_volume,((0,output_shape[0]-input_shape[0]),(0,output_shape[1]-input_shape[1]),(0,output_shape[2]-input_shape[2])),mode= 'constant')
    input_mask = np.pad(mask,((0,output_shape[0]-input_shape[0]),(0,output_shape[1]-input_shape[1]),(0,output_shape[2]-input_shape[2])),mode= 'constant')

    # score_map =get_scoremap3D(window_size=window_size)
    class_matrix = np.zeros([output_index[0]+1,output_index[1]+1,output_index[2]+1,num_labels])

    # 对输入3D图像进行滑动窗口推理
    for i in range(output_index[0]+1):
        for j in range(output_index[1]+1):
            for k in range(output_index[2]+1):
                # 提取当前窗口的数据
                x = input_volume[i*step_size[0]:i*step_size[0]+window_size[0],
                                j*step_size[1]:j*step_size[1]+window_size[1],
                                k*step_size[2]:k*step_size[2]+window_size[2]]
                m = input_mask[i*step_size[0]:i*step_size[0]+window_size[0],
                                j*step_size[1]:j*step_size[1]+window_size[1],
                                k*step_size[2]:k*step_size[2]+window_size[2]]
    
                x = x[np.newaxis,np.newaxis,:,:,:]
                m = m[np.newaxis,np.newaxis,:,:,:]


                x = torch.Tensor(x).to(device)
                m = torch.Tensor(m).to(device)
                #print('input',x.shape)
                # 使用模型进行预测
                if addmask:
                    x = torch.cat([x,m],dim=1)
                y = model(x)
                y =  torch.sigmoid(y)
                # 将预测结果转换为NumPy数组，并调整维度顺序
                y = y.squeeze().detach().cpu().numpy()
                # 将预测结果保存到输出3D图像中
                class_matrix[i,j,k,:] =y

    # ##平均值：
    # output = output_volume  /inferce_map

    # output = output[:input_shape[0],:input_shape[1],:input_shape[2]]

    return class_matrix