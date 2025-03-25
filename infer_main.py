import nibabel as nib
import numpy as np 
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
import SimpleITK as sitk
import torch 
from image_process import crop_edge_sample,crop_pad3D
from monai.inferers import sliding_window_inference
from class_tool import class_inference
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def load_nifit(data_path):
    img = nib.load(data_path)
    tmp = np.squeeze(img.get_fdata()).astype(np.float32)
    return tmp


def save_nifit(data, filename):
    # print(data.dtype)
    img = nib.Nifti1Image(data, np.eye(4))
    nib.save(img, filename)

def norm_CT(X,win_min=0,win_max=200):
    X[X<win_min]=win_min
    X[X>win_max]=win_max
    X = (X-win_min)/(win_max - win_min)
    return X
def resampleVolume(outspacing, vol,resamplemethod=sitk.sitkLinear):
    """
    将体数据重采样的指定的spacing大小\n
    paras：
    outpacing：指定的spacing，例如[1,1,1]
    vol：sitk读取的image信息，这里是体数据\n
    return：重采样后的数据
    """
    outsize = [0, 0, 0]
    # 读取文件的size和spacing信息
    inputsize = vol.GetSize()
    inputspacing = vol.GetSpacing()
 
    transform = sitk.Transform()
    transform.SetIdentity()
    # 计算改变spacing后的size，用物理尺寸/体素的大小
    outsize[0] = round(inputsize[0] * inputspacing[0] / outspacing[0])
    outsize[1] = round(inputsize[1] * inputspacing[1] / outspacing[1])
    outsize[2] = round(inputsize[2] * inputspacing[2] / outspacing[2])
 
    # 设定重采样的一些参数
    resampler = sitk.ResampleImageFilter()
    resampler.SetTransform(transform)
    resampler.SetInterpolator(resamplemethod)
    resampler.SetOutputOrigin(vol.GetOrigin())
    resampler.SetOutputSpacing(outspacing)
    resampler.SetOutputDirection(vol.GetDirection())
    resampler.SetSize(outsize)
    newvol = resampler.Execute(vol)
    return newvol
def prepare_data(data_path=None):
    data = sitk.ReadImage(data_path)
    data_resize = resampleVolume(outspacing=[1,1,1],vol=data)
    data_array = sitk.GetArrayFromImage(data_resize)
    return data_array


def infer(file_path = '/data/yunzhixu/Data/CVST/test_data/zhouguiqing.nii.gz',
          resize_path = '/data/yunzhixu/Data/CVST/test_data/zhouguiqing_resize.nii.gz',
          seg_path='/data/yunzhixu/Data/CVST/test_data/zhouguiqing_resize_pred.nii.gz',
          mask_path='/data/yunzhixu/Data/CVST/test_data/zhouguiqing_resize_maskdata.nii.gz',
          mask_seg_path ='/data/yunzhixu/Data/CVST/test_data/zhouguiqing_resize_mask.nii.gz',
          save=True,
          label_mask_path =None,
          seg_model =None ,
          class_model = None):##用于使用真实的mask来进行测试全图
    ##seg:
    CT_data = prepare_data(file_path)
    if label_mask_path !=None:
        label_mask =  sitk.GetArrayFromImage(resampleVolume([1,1,1],sitk.ReadImage(label_mask_path),sitk.sitkNearestNeighbor))
    if save:
        save_nifit(CT_data,resize_path)
    # print('CT_data',CT_data.shape)
    if label_mask_path !=None:
        pred_array = label_mask
    else:
        patch_size =[128,128,128]
        # model = torch.load('/data/yunzhixu/CVST_test/segmentation/train_class_CVST_tseg_p128_UNetCBAM_augdrop_diceceloss_train_file_fold_total_n32_pretrain.pth').cuda()
        # model1 = torch.load('/data/yunzhixu/CVST_test/segmentation/train_class_CVST_tseg_p128_dice_augdrop_diceceloss_trainfile_fold_total_n32_pretrain.pth').cuda()
        # model = torch.load('/data/yunzhixu/CVST_test/segmentation/train_class_CVST_tseg_p128_UNetCBAMPlus_augdrop_diceceloss_fold_total_n32.pth').cuda()
        model = seg_model

        image =norm_CT(CT_data.copy())
        image = crop_pad3D(image,[256,256,256])
        # print(CT_data.shape)
        model.eval()
        # model1.eval()
        # model.eval()
        with torch.no_grad():
            x = image[np.newaxis,np.newaxis,:,:,:]
            x = torch.Tensor(x).to(device)
            pred = sliding_window_inference(x, (patch_size[0], patch_size[1], patch_size[2]), 4, model)
            # pred1 = sliding_window_inference(x, (patch_size[0], patch_size[1], patch_size[2]), 4, model1)
            pred = torch.sigmoid(pred)
            # pred1 = torch.sigmoid(pred1)
            # pred = pred*0.5 + pred1 *0.5


            # ###data_aug_pred
            # pred_shift,pred_flip0,pred_flip1,pred_flip2 = data_aug_pred(x,model)
            # pred = 0.3*pred +0.1*pred_shift+0.2*pred_flip0+0.2*pred_flip1+0.2*pred_flip2

            # pred_shift,pred_flip0,pred_flip1,pred_flip2 = data_aug_pred(x,model1)
            # pred1 = 0.3*pred1 +0.1*pred_shift+0.2*pred_flip0+0.2*pred_flip1+0.2*pred_flip2

            # pred = 0.5*pred + 0.5*pred1
            # x_shift = torch.roll(x,shifts=[30,30,30],dims=(2,3,4))
            # pred_shift = sliding_window_inference(x_shift, (patch_size[0], patch_size[1], patch_size[2]), 4, model)
            # pred_shift = torch.roll(pred_shift,shifts= [-30,-30,-30],dims=(2,3,4))
            # pred_shift = torch.sigmoid(pred_shift)
            # x_flip = x.cpu().numpy()
            # # print('x_flip',x_flip.shape)
            # x_flip  = torch.Tensor(x_flip[:,:,::-1,::-1,:].copy()).cuda()
            # pred_flip = sliding_window_inference(x_flip, (patch_size[0], patch_size[1], patch_size[2]), 4, model)
            # pred_flip = torch.Tensor(pred_flip.cpu().numpy()[:,:,::-1,::-1,:].copy()).cuda()
            # pred_flip = torch.sigmoid(pred_flip)

            # x_rot = torch.rot90(x,k=1,dims=(2,3))
            # pred_rot = sliding_window_inference(x_rot, (patch_size[0], patch_size[1], patch_size[2]), 4, model)
            # pred_rot = torch.rot90(pred_rot,k=-1,dims=(2,3))



            # pred = 0.3*pred +0.25*pred_shift+0.25*pred_flip+0.2*pred_rot


            pred[pred<0.5]=0
            pred[pred>0.5]=1
        
        pred_array = pred.cpu().numpy()[0,0,:,:,:]
        if save:
            save_nifit(pred_array,seg_path)
        # print('seg array sum',np.sum(pred_array))
    if np.sum(pred_array)==0:
        print('未分割出区域')
        infer_result = 0
    else:

    ##取mask:
        data_array = CT_data
        mask_array = pred_array
        X_final,Y_final,L,index = crop_edge_sample(mask_array[np.newaxis,:,:,:],data_array[np.newaxis,:,:,:],resample=None)
        index =index[0]
        center = [int((index[0]+index[1])/2),int((index[2]+index[3])/2),int((index[4]+index[5])/2)]
        # print(center)
        X_final = np.array(X_final)[0]
        Y_final = np.array(Y_final)[0]
        patch_size = [128,128,128]    ###修改patchsize,取一大部分
        start_pos = [0,0,0]
        end_pos = [0,0,0]
        # print('index',index )
        for i in range(3):
            if (center[i] -  patch_size[i]/2)<0:
                start_pos[i] = 0
                end_pos[i] = patch_size[i]
            else:
                start_pos[i] = int(center[i] - patch_size[i]/2)
                end_pos[i] = int(center[i] + patch_size[i]/2)
            if (center[i] +  patch_size[i]/2)>data_array.shape[i]:
                end_pos[i] = data_array.shape[i]
                start_pos[i] = end_pos[i]-patch_size[i]


        # print(start_pos,end_pos)
        mask_array = mask_array[start_pos[0]:end_pos[0],start_pos[1]:end_pos[1],start_pos[2]:end_pos[2]]
        data_array = data_array[start_pos[0]:end_pos[0],start_pos[1]:end_pos[1],start_pos[2]:end_pos[2]]

        if save:
            save_nifit(data_array,mask_path)
            save_nifit(mask_array,mask_seg_path)





        
        # print('data_array',data_array.shape)
        if data_array.shape[0]<patch_size[0] or data_array.shape[1]<patch_size[1] or data_array.shape[2]<patch_size[2]:
            data_array =crop_pad3D(data_array,patch_size)
            mask_array = crop_pad3D(mask_array,patch_size)
        # del model

        model = class_model

        infer_result =class_inference(norm_CT(data_array),model,is_patch=True,mask=mask_array,addmask=True,augpred=False,multi_class=False,patch_size=[128,128,128])##muti_class =False为2分类
        # infer_result = np.mean(infer_result)
        ##class dataaug:
        x = norm_CT(data_array)
        mask_aug = mask_array

        x_shift = np.roll(x,10,axis=(0,1,2))
        mask_shift = np.roll(mask_aug,10,axis=(0,1,2))
        infer_result_shift =class_inference(x_shift,model,is_patch=True,mask=mask_shift,addmask=True,augpred=False,multi_class=False,patch_size=[128,128,128])

        x_flip0  = x[::-1,:,:].copy()
        mask_flip0= mask_aug[::-1,:,:]
        infer_result_flip0 =class_inference(x_flip0,model,is_patch=True,mask=mask_flip0,addmask=True,augpred=False,multi_class=False,patch_size=[128,128,128])
        # x_flip1  = x[:,::-1,:].copy()
        # mask_flip1= mask_aug[:,::-1,:]
        # infer_result_flip1 =class_inference(x_flip1,model,is_patch=True,mask=mask_flip1,addmask=True,augpred=False,multi_class=False,patch_size=[128,128,128])
        # x_flip2  = x[:,:,::-1].copy()
        # mask_flip2= mask_aug[:,:,::-1]
        # infer_result_flip2 =class_inference(x_flip2,model,is_patch=True,mask=mask_flip2,addmask=True,augpred=False,multi_class=False,patch_size=[128,128,128])


        x_rot = np.rot90(x,1,(1,2)).copy()
        mask_rot = np.rot90(mask_aug,1,(1,2))
        infer_result_rot =class_inference(x_rot,model,is_patch=True,mask=mask_rot,addmask=True,augpred=False,multi_class=False,patch_size=[128,128,128])
        infer_result_origin = infer_result.copy()
        infer_result = 0.3*infer_result +0.2*infer_result_shift +0.3*infer_result_flip0+0.2*infer_result_rot
        # infer_result = 0.2*infer_result +0.2*infer_result_flip0 + 0.1*infer_result_flip1 +0.1*infer_result_flip2 + 0.2*infer_result_shift + 0.2*infer_result_rot



    return infer_result_origin,infer_result


def infer_case(test_path,pred_save_path='/',test_name='test0',save_file=False):
    '''
    用于直接测试新增的数据
    '''




    # model = UNet_nest3_3d_CBAM_plus_ED(n1=32,bn=True,num_classes=1,avg=[8,8,8],in_channels=2).to(device)
    # model = UNet_nest3_3d_ED(n1=32,bn=True,num_classes=1,avg=[8,8,8],in_channels=2).to(device)




    # seg_model = UNet_nest3_3d(in_channels=1,
    #         out_channels=1,
    #         n1=32,
    #         bn=True).cuda()
    # seg_model.load_state_dict(torch.load(seg_model_path[0]).state_dict())
    # class_model =  UNet_nest3_3d_ED(n1=32,bn=True,num_classes=1,avg=[8,8,8],in_channels=2).cuda()
    # class_model.load_state_dict(torch.load(class_model_path[0]).state_dict())



 
    # torch.jit.load("./model_name.pth")
    
    seg_model_path = ["train_class_CVST_tseg_p128_dice_augdrop_diceceloss_trainfile_fold_total_n32_pretrain_jit.pth"]
    class_model_path = ['train_class_CVST_maskpatch_foldtotal_segweight_addmask_nofreeze_035PICH_bestTruenum_bestauc_jit.pth',
                        'train_class_CVST_maskpatch_UNetCBAM_foldtotal_segweight_nofreeze_035PICH_bestauc_jit.pth']

    seg_model = torch.jit.load(seg_model_path[0],map_location=device).to(device)


    pred_name = ['UNet     ','UNet+CBAM']
    oth = [0.5,0.9244]
    for i in range(2):
        class_model = torch.jit.load(class_model_path[i],map_location=device).to(device)

        # test_name = '' 
        # test_path = ''
        # pred_save_path = ''

        
        tmp,tmp_aug = infer(file_path = str(test_path),
                    resize_path = str(pred_save_path+str(test_name)+'_resize.nii.gz'),
                    seg_path=str(pred_save_path+str(test_name)+'_resize_pred.nii.gz'),
                    mask_path=str(pred_save_path+str(test_name)+'_resize_maskdata.nii.gz'),
                    mask_seg_path =str(pred_save_path+str(test_name)+'_resize_mask.nii.gz'),
                    save=save_file,
                    seg_model=seg_model,
                    class_model=class_model)
        if tmp<oth[i]:
            pred_case ='CVST-ICH'
        else:
            pred_case = 'sICH'
        print(pred_name[i],'=',tmp,str(tmp_aug)+' pred '+str(pred_case))


import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='run inference')
    parser.add_argument("--file", type=str, default="0")
    args = parser.parse_args()
    if args.file == "0":
        file_path = input('输入测试Nifit文件路径:')
        infer_case(test_path=str(file_path))
    else:
        file_path = args.file
        infer_case(test_path=str(file_path))
    


