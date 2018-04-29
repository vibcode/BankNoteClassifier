import os
import numpy as np
import scipy
from scipy import ndimage
import pickle

count=0
train_dataset=np.ndarray((540,130,258),dtype=np.float32)
train_labels=np.ndarray((540),dtype=np.int32)
folders=['10','20','50','100','500','2000']

'''
def vectorise(folder):
    a=np.zeros((1,6),dtype=np.float32)
    if(folder=='10'):
        a[0,0]=1.0
    elif(folder=='20'):
        a[0,1]=1.0
    elif(folder=='50'):
        a[0,2]=1.0
    elif(folder=='100'):
        a[0,3]=1.0
    elif(folder=='500'):
        a[0,4]=1.0
    elif(folder=='2000'):
        a[0,5]=1.0
    return a

'''

flag=0
for folder in folders:
    print folder
    images=os.listdir(folder)
    for image in images:
        resizedi=np.ndarray((130,258),dtype=np.float32)
        imagepath=os.path.join(folder,image)
        i=ndimage.imread(imagepath,flatten=1).astype(float)

        resizedi=scipy.misc.imresize(i,(130,258))
        train_dataset[count,:,:]=resizedi

        if(folder=='10'):
            train_labels[count]=10
        elif(folder=='20'):
            train_labels[count]=20
        elif(folder=='50'):
            train_labels[count]=50
        elif(folder=='100'):
            train_labels[count]=100
        elif(folder=='500'):
            train_labels[count]=500
        elif(folder=='2000'):
            train_labels[count]=2000
        count+=1

dict={
'train_dataset': train_dataset,
'train_labels': train_labels,
}

with open('banknote1.pickle','wb') as f:
    pickle.dump(dict,f)


#errrors may be due to
#large size of input image
#histogram implementation use library
#small dataset
#tuning gamma and C parameter
