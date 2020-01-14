import os
import SimpleITK as sitk
import numpy as np

if __name__ == '__main__':
    url = '/home/hym/Documents/yankai/dataset/all_label'
    names = os.listdir(url)
    for name in names:
        print(name)
        path = url + '/' + name
        image = sitk.ReadImage(path)
        dicom_array = sitk.GetArrayFromImage(image)
        # origin = np.array(image.GetOrigin())
        # spacing = np.array(image.GetSpacing())
        print('shape',dicom_array.shape)
        # print(origin,spacing)
        # aaa = np.zeros(dicom_array.shape)
        # aaa[dicom_array ==3 ] =1
        # sitk.WriteImage(sitk.GetImageFromArray(aaa),'adafda.vtk')   #标签读得没毛病
        # break