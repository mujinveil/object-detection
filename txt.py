#coding utf-8 
import os 
import random 

xmlfilepath='./VOC2020/Annotations'  
saveBasePath='./VOC2020'  
trainval_percent=1  # 划分整个训练集百分之几作为 trainval
train_percent=1    # trainval 中  train 所占比例

total_xml = os.listdir(xmlfilepath)
num=len(total_xml)   
list=range(num) 
tv=int(num*trainval_percent)    
tr=int(tv*train_percent) 

trainval= random.sample(list,tv)    
train=random.sample(trainval,tr) 

print("train and val size",tv)  
print("traub suze",tr)  
ftrainval = open(os.path.join(saveBasePath,'ImageSets/Main/trainval.txt'), 'w')  
ftest = open(os.path.join(saveBasePath,'ImageSets/Main/test.txt'), 'w')
ftrain = open(os.path.join(saveBasePath,'ImageSets/Main/train.txt'), 'w') 
fval = open(os.path.join(saveBasePath,'ImageSets/Main/val.txt'), 'w')   


for i  in list:  
    name=total_xml[i][:-4]+'\n'  
    if i in trainval:   
        ftrainval.write(name)   
        if i in train:   
            ftrain.write(name)    
        else:   
            fval.write(name)  
    else:    
        ftest.write(name)  

ftrainval.close()    
ftrain.close()  
fval.close()    
ftest .close()  
