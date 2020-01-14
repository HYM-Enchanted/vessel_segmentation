import os
import SimpleITK as sitk
import numpy as np
import load_data
import torch
import copy


if __name__ == '__main__':
    # url = '/home/hym/Documents/vessel_data/train_data1'
    url = '/home/hym/hym/vessel_train_data'
    names = os.listdir(url)
    wei_all = 0
    model_save_path = '/home/hym/Documents/vtk'
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    nums = len(names)
    for name in names:
        print(name)
        path = os.path.join(url,name) + '/original1'
        image_array, origin, spacing = load_data.load_dicom(path)
        # s_z,s_x,s_y = image_array.shape
        # print(name,'\n',image_array.shape,'\n',spacing)
        mask_path = os.path.join(url,name) + '/vein'
        a,b,c = load_data.load_dicom(mask_path)
        print('aaaaaa',sum(sum(sum(a))))
        d = image_array*a

        print(type(d))
        print('max min',np.min(d),np.max(d))
        print('/////',np.min(image_array),np.max(image_array))
        d[d < 0] = 0
        d[d > 0] = 1
        print('ddddd',sum(sum(sum(d))))
        # print(sum(sum(sum(a))))
        #将标签按照阈值截断之后写成vtk格式check
        min_bound = 0
        max_bound = 900
        bbb = copy.deepcopy(image_array)
        bbb[bbb < min_bound] = 0
        bbb[bbb > max_bound] = 900
        # print('bnbbbbbb',np.min(bbb),np.max(bbb))
        if (bbb == image_array).all():
            print('222222222')
        eee = bbb * a
        eee[eee < 0] = 0
        eee[eee > 0] = 1
        print('eeeee',sum(sum(sum(eee))))
        if (d ==eee).all():
            print('111111111')
        # qq = sitk.GetImageFromArray(eee)
        # print("okokokk~~~~~")
        # sitk.WriteImage(qq, model_save_path +'/'+ str(name) + '.vtk')


    #     wei = np.mean(a)
    #     print('weight',wei)
    #     wei_all += wei
    #     box = a.shape
    #     sum_object = box[0]*box[1]*box[2] * wei
    #     print('sada',box[0]*box[1]*box[2])
    #     print('sada',box[0],box[1],box[2])
    #     print('sadadad',sum_object)
    #     break
    #
    # final_weight = 1/((wei_all / nums))
    # print('final',final_weight)
    # weight1 = final_weight * 3
    # weight2 = 1/weight1
    # nll_weight = torch.tensor([weight2,1-weight2])
    # print(nll_weight)