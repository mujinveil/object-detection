#coding utf-8 
import sys
import numpy as np
import logging
from model import *
from ssd import *
''''
start_iter=int(sys.argv[1])
epochs=int(sys.argv[2])
lr = float(sys.argv[3])
finetune = int(sys.argv[4])
'''
start_iter = 3
epochs =2
lr=0.2
finetune=-1

model_name='ssd'
mx.random.seed(1)
ctx=mx.cpu(1)
if finetune ==1:
    net = ToySSD(1)
    net.load_params(model_name+'_%d.params' % (17999),ctx)
    train(net, epochs=epochs,lr=lr,start_epoch=start_iter,ctx=ctx)
elif finetune ==-1:
    net = ToySSD(1)
    net.load_params(model_name+'_%d.params' % (4),ctx)
    test(net,ctx)
    '''
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        test(net,ctx=ctx,image=frame)
        #cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    '''
else:
    net = ToySSD(1)
    net.initialize(mx.init.Xavier(magnitude=2), ctx=ctx)
    net.collect_params().reset_ctx(ctx)
    train(net, epochs=epochs,lr=lr,start_epoch=start_iter,ctx=ctx)
