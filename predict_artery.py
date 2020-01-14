import torch
import vnet
import SimpleITK as sitk
import train_artery
import numpy as np
import load_data

def make_predict_data(url):
    image = sitk.ReadImage(url)   #读dicom
    dicom_array = sitk.GetArrayFromImage(image)
    # origin = np.array(image.GetOrigin())
    # spacing = np.array(image.GetSpacing())
    dicom_array = train_artery.normalize_data(dicom_array)
    size_z,size_x,size_y = dicom_array.shape
    url_mask = url.replace('volume', 'segmentation')
    url_mask = url_mask.replace('all', 'all_label')
    mask = sitk.ReadImage(url_mask)  #读标签
    mask_array = sitk.GetArrayFromImage(mask)
    maskkk = np.zeros(mask_array.shape)
    maskkk[mask_array == 3] = 1   #标签中标记为3的才是动脉
    # print(sum(sum(sum(maskkk))))
    i_crop = load_data.crop(dicom_array,mode='test')  #crop
    m_crop = load_data.crop(maskkk,mode='test')
    result = []
    for k in range(len(i_crop)):
        ik = i_crop[k][np.newaxis, :]
        mk = m_crop[k][np.newaxis, :]
        result.append((ik, mk))
    print('测试数据有{}个batch'.format(len(result)))
    return result,size_z,size_x,size_y

if __name__ == '__main__':
    url = '/home/hym/Documents/yankai/dataset/all/volume-8.nii'
    pred_data,z,x,y = make_predict_data(url)
    print('size',x,y,z)
    pred_res = torch.zeros(z,x,y)
    nn = 0

    # model = vnet.VNet(elu=False, nll=False)
    # model.load_state_dict(torch.load('artery_params_1/vnet_params11.pkl'))
    # model.load_state_dict(torch.load('artery_params_3/vnet_params9.pkl'))
    # model.load_state_dict(torch.load('artery_params_8/vnet_params44.pkl'))
    model = torch.load('artery_params_9my_model3.pth')
    model.cuda()
    model.eval()  #测试模式

    for batch_i,xy in enumerate(pred_data):
        # print('ababdbada',batch_i)
        data_x = xy[0][np.newaxis,:]
        # data_y = xy[1]
        image = torch.from_numpy(data_x)
        image = image.type(torch.FloatTensor).cuda()
        # target = torch.from_numpy(data_y)
        # target = target.type(torch.FloatTensor).cuda()
        # target = target.view(target.numel())
        with torch.no_grad():
            output = model(image)  # n*2维
            _, pred = output.data.max(1)  # 预测点，第1维是背景0，第二维是前景1
            re_pred = torch.reshape(pred, (4, 208, 336))
            if nn + 4 < z:
                # print('nn',nn)
                pred_res[nn:nn+ 4, 152:360, 88:424] = re_pred
                nn += 4
            else:
                pred_res[-4:, 152:360, 88:424] = re_pred

    final = pred_res
    sum_pred = torch.sum(final)
    print('sum predict dots', sum_pred)
    # sitk.WriteImage(sitk.GetImageFromArray(final),'predict-100.vtk')


    maskurl = '/home/hym/Documents/yankai/dataset/all_label/segmentation-8.nii'
    mask = sitk.ReadImage(maskurl)
    mask_array = sitk.GetArrayFromImage(mask)
    maskkk = np.zeros(mask_array.shape)
    maskkk[mask_array == 3] = 1
    sum_mask = np.sum(maskkk)
    print('mask sum',sum_mask)
    pred_ = final.numpy()
    print(pred_.shape,maskkk.shape)
    inters = pred_ * maskkk
    print('intersect',sum(sum(sum(inters))))
    # maskkk = torch.from_numpy(maskkk)
    # intersect = torch.dot(torch.squeeze(final), torch.squeeze(maskkk))
    # print('intersect',intersect)