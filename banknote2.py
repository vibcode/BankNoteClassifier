import pickle
import numpy as np
import math

def grad(image):
    gradient=np.ndarray((128,256),dtype=np.float32)
    gmagnitude=np.ndarray((128,256),dtype=np.float32)
    for i in range(1,129):
        for j in range(1,257):
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

with open('banknote1.pickle') as f:
    dict=pickle.load(f)
    train_dataset=dict['train_dataset']
    train_labels=dict['train_labels']

hogfeatures=np.ndarray((540,3780),dtype=np.float32)


for i in range(0,540):
    print i
    gradient=np.ndarray((128,256),dtype=np.float32)
    gmagnitude=np.ndarray((128,256),dtype=np.float32)
    image=np.ndarray((130,258),dtype=np.float32)
    image=train_dataset[i,:,:]
    gradient,gmagnitude=grad(image)

    j=0
    k=0
    count=0
    while(j<=96):
        k=0
        while(k<=224):
            histcell1=np.ndarray((1,9),dtype=np.float32)
            histcell2=np.ndarray((1,9),dtype=np.float32)
            histcell3=np.ndarray((1,9),dtype=np.float32)
            histcell4=np.ndarray((1,9),dtype=np.float32)
            blockhist=np.ndarray((1,36),dtype=np.float32)
            histcell1=histogram(gradient,gmagnitude,j,j+15,k,k+15)
            histcell2=histogram(gradient,gmagnitude,j,j+15,k+16,k+31)
            histcell3=histogram(gradient,gmagnitude,j+16,j+31,k,k+15)
            histcell4=histogram(gradient,gmagnitude,j+16,j+31,k+16,k+31)
            blockhist[0,:9]=histcell1
            blockhist[0,9:18]=histcell2
            blockhist[0,18:27]=histcell3
            blockhist[0,27:]=histcell4
            blockhist=normalise(blockhist)
            hogfeatures[i,count:count+36]=blockhist
            count+=36
            k+=16
        j+=16

dict['svmhogfeatures']=hogfeatures
dict['train_labels']=train_labels
dict['train_dataset']=train_dataset

with open('svmhogfeatures.pickle','wb') as f:
    pickle.dump(dict,f)
