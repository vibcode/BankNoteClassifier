from sklearn import svm
import os
import numpy as np
import scipy
from scipy import ndimage
import pickle
import math

dict={}
with open('svmhogfeatures.pickle','rb') as f:
    dict=pickle.load(f)
    svmhogfeatures=dict['svmhogfeatures']
    train_labels=dict['train_labels']
    train_dataset=dict['train_dataset']


per=np.random.permutation(train_labels.shape[0])
svmhogfeatures=svmhogfeatures[per,:]
train_labels=train_labels[per]


classifier=svm.SVC()
classifier.fit(svmhogfeatures,train_labels)



def grad(image):
    gradient=np.ndarray((400,800),dtype=np.float32)
    gmagnitude=np.ndarray((400,800),dtype=np.float32)
    for i in range(1,401):
        for j in range(1,801):
            gx=image[i-1,j]-image[i+1,j]
            gy=image[i,j-1]-image[i,j+1]
            if(gx==0 and gy==0):
                gmagnitude[i-1,j-1]=0
                gradient[i-1,j-1]=0
            elif(gx==0):
                gmagnitude[i-1,j-1]=abs(gy)
                gradient[i-1,j-1]=90
            elif(gy==0):
                gmagnitude[i-1,j-1]=abs(gx)
                gradient[i-1,j-1]=0
            else:

                gmagnitude[i-1,j-1]=math.sqrt(gx*gx+gy*gy)
                a=math.degrees(math.atan(gy/gx))
                if(a>0):
                    gradient[i-1,j-1]=a
                else:
                    gradient[i-1,j-1]=a+180

    return gradient,gmagnitude

def histogram(gradient,gmagnitude,j1,j2,k1,k2):
    hist= np.zeros((1,9),dtype=np.float32)
    for i in range(j1,j2+1):
        for j in range(k1,k2+1):
            if(gradient[i,j]>=0 and gradient[i,j]<=20):
                hist[0,0]+=gmagnitude[i,j]*(20-gradient[i,j])/20
                hist[0,1]+=gmagnitude[i,j]*(gradient[i,j]-0)/20
            elif(gradient[i,j]>20 and gradient[i,j]<=40):
                hist[0,1]+=gmagnitude[i,j]*(40-gradient[i,j])/20
                hist[0,2]+=gmagnitude[i,j]*(gradient[i,j]-20)/20
            elif(gradient[i,j]>40 and gradient[i,j]<=60):
                hist[0,2]+=gmagnitude[i,j]*(60-gradient[i,j])/20
                hist[0,3]+=gmagnitude[i,j]*(gradient[i,j]-40)/20
            elif(gradient[i,j]>60 and gradient[i,j]<=80):
                hist[0,3]+=gmagnitude[i,j]*(80-gradient[i,j])/20
                hist[0,4]+=gmagnitude[i,j]*(gradient[i,j]-60)/20
            elif(gradient[i,j]>80 and gradient[i,j]<=100):
                hist[0,4]+=gmagnitude[i,j]*(100-gradient[i,j])/20
                hist[0,5]+=gmagnitude[i,j]*(gradient[i,j]-80)/20
            elif(gradient[i,j]>100 and gradient[i,j]<=120):
                hist[0,5]+=gmagnitude[i,j]*(120-gradient[i,j])/20
                hist[0,6]+=gmagnitude[i,j]*(gradient[i,j]-100)/20
            elif(gradient[i,j]>120 and gradient[i,j]<=140):
                hist[0,6]+=gmagnitude[i,j]*(140-gradient[i,j])/20
                hist[0,7]+=gmagnitude[i,j]*(gradient[i,j]-120)/20
            elif(gradient[i,j]>140 and gradient[i,j]<=160):
                hist[0,7]+=gmagnitude[i,j]*(160-gradient[i,j])/20
                hist[0,8]+=gmagnitude[i,j]*(gradient[i,j]-140)/20
            elif(gradient[i,j]>160 and gradient[i,j]<=180):
                hist[0,8]+=gmagnitude[i,j]*(180-gradient[i,j])/20
                hist[0,0]+=gmagnitude[i,j]*(gradient[i,j]-160)/20

    return hist
def normalise(blockhist):
    norm=0
    for i in range(0,36):
        norm+=(blockhist[0,i])*(blockhist[0,i])
    norm=math.sqrt(norm)
    for i in range(0,36):
        blockhist[0,i]=(blockhist[0,i])/norm
    return blockhist
'''
def sroot(im):
    for i in range(0,258):
	for j in range(0,130):
            im[i,j]=math.sqrt(im[i,j])

    return im
'''

def check(folder):
    if(folder=='t10'):
        return 10
    if(folder=='t20'):
        return 20
    if(folder=='t50'):
        return 50
    if(folder=='t100'):
        return 100
    if(folder=='t500'):
        return 500
    if(folder=='t2000'):
        return 2000

#testing with new image
folders=['t10','t20','t50','t100','t500','t2000',]
counter=0
for folder in folders:
    note=check(folder)
    images=os.listdir(folder)
    images.sort()
    for image in images:
        imagepath=os.path.join(folder,image)
        test_dataset=np.ndarray((1,402,802),dtype=np.float32)
        resizedimage=np.zeros((402,802),dtype=np.float32)
        imag=ndimage.imread(imagepath,flatten=1).astype(float)
        resizedimage=scipy.misc.imresize(imag,(402,802))

        test_dataset[0,:,:]=resizedimage

        gradient=np.ndarray((400,800),dtype=np.float32)
        gmagnitude=np.ndarray((400,800),dtype=np.float32)
        imge=np.ndarray((402,802),dtype=np.float32)
        imge=test_dataset[0,:,:]

        gradient,gmagnitude=grad(imge)
        hogfeatures=np.ndarray((1,3780),dtype=np.float32)

        j=0
        k=0
        count=0
        while(j<=300):
            k=0
            while(k<=700):
                histcell1=np.ndarray((1,9),dtype=np.float32)
                histcell2=np.ndarray((1,9),dtype=np.float32)
                histcell3=np.ndarray((1,9),dtype=np.float32)
                histcell4=np.ndarray((1,9),dtype=np.float32)
                blockhist=np.ndarray((1,36),dtype=np.float32)
                histcell1=histogram(gradient,gmagnitude,j,j+49,k,k+49)
                histcell2=histogram(gradient,gmagnitude,j,j+49,k+50,k+99)
                histcell3=histogram(gradient,gmagnitude,j+50,j+99,k,k+49)
                histcell4=histogram(gradient,gmagnitude,j+50,j+99,k+50,k+99)
                blockhist[0,:9]=histcell1
                blockhist[0,9:18]=histcell2
                blockhist[0,18:27]=histcell3
                blockhist[0,27:]=histcell4
                blockhist=normalise(blockhist)
                hogfeatures[0,count:count+36]=blockhist
                count+=36
                k+=50
            j+=50


        print 'Actual-',note,' prediction-',classifier.predict(hogfeatures)[0]
