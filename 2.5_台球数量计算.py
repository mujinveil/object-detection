#coding utf-8 
import time,os
from matplotlib import pyplot as plt 
import numpy as np 
import mxnet as mx 
from mxnet import autograd,gluon,init
import gluoncv as gcv 
from gluoncv.utils import download,viz 
from gluoncv import data as gdata
from gluoncv.utils.metrics.voc_detection import VOC07MApMetric
from gluoncv.data.transforms.presets.ssd import SSDDefaultTrainTransform
from gluoncv.data.transforms.presets.ssd import SSDDefaultValTransform


train_dataset=gdata.VOCDetection(splits=[(2020,'trainval')])
valid_dataset=gdata.VOCDetection(splits=[(2020,'test')])
net=gcv.model_zoo.get_model('ssd_300_vgg16_atrous_voc',pretrained=False)
classes=['ball']
net.reset_class(classes)

ctx=[mx.cpu(0)]
net=gcv.model_zoo.get_model('ssd_300_vgg16_atrous_custom',classes=classes,pretrained_base=False,transfer='voc')
net.load_parameters('ssd_300_vgg16_ball_v2.params')
#net.initialize(init.Xavier(),ctx=ctx)



def get_dataloader(net,train_dataset,valid_dataset,data_shape,batch_size,num_workers):
    from gluoncv.data.batchify import Tuple,Stack,Pad
    from gluoncv.data.transforms.presets.ssd import SSDDefaultTrainTransform
    width,height=data_shape,data_shape
    # use fake data to generate fixed anchors for target generation
    with autograd.train_mode():
        _, _, anchors = net(mx.nd.zeros((1, 3, height, width)))
    batchify_fn = Tuple(Stack(), Stack(), Stack())  # stack image, cls_targets, box_targets
    train_loader = gluon.data.DataLoader(
        train_dataset.transform(SSDDefaultTrainTransform(width, height, anchors)),
        batch_size, True, batchify_fn=batchify_fn, last_batch='rollover', num_workers=num_workers)

    val_batchify_fn=Tuple(Stack(),Pad(pad_val=-1))

    val_loader =gluon.data.DataLoader(
        valid_dataset.transform(SSDDefaultValTransform(width,height)),
        batch_size,False,batchify_fn=val_batchify_fn,last_batch='keep',num_workers=num_workers)

    eval_metric = VOC07MApMetric(iou_thresh=0.5, class_names=classes)
    return train_loader,val_loader,eval_metric


def train():
    net.collect_params().reset_ctx(ctx)
    trainer = gluon.Trainer(
        net.collect_params(), 'sgd',
        {'learning_rate': 0.001, 'wd': 0.0005,'momentum': 0.9})

    mbox_loss = gcv.loss.SSDMultiBoxLoss()
    ce_metric = mx.metric.Loss('CrossEntropy')
    smoothl1_metric = mx.metric.Loss('SmoothL1')

    for epoch in range(0,20):
        ce_metric.reset()
        smoothl1_metric.reset()
        tic = time.time()
        btic = time.time()
        net.hybridize(static_alloc=True, static_shape=True)
        for i, batch in enumerate(train_data):
            batch_size = batch[0].shape[0]
            data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
            cls_targets = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0)
            box_targets = gluon.utils.split_and_load(batch[2], ctx_list=ctx, batch_axis=0)
            with autograd.record():
                cls_preds = []
                box_preds = []
                for x in data:
                    cls_pred, box_pred, _ = net(x)
                    cls_preds.append(cls_pred)
                    box_preds.append(box_pred)
                sum_loss, cls_loss, box_loss = mbox_loss(
                    cls_preds, box_preds, cls_targets, box_targets)

                autograd.backward(sum_loss)
            # since we have already normalized the loss, we don't want to normalize
            # by batch-size anymore
            trainer.step(1)
            ce_metric.update(0, [l * batch_size for l in cls_loss])
            smoothl1_metric.update(0, [l * batch_size for l in box_loss])
            name1, loss1 = ce_metric.get()
            name2, loss2 = smoothl1_metric.get()
            if i % 10 == 0:
                print('[Epoch {}][Batch {}], Speed: {:.3f} samples/sec, {}={:.3f}, {}={:.3f}'.format(
                    epoch, i, batch_size/(time.time()-btic), name1, loss1, name2, loss2))
            if i%10==0:
                net.save_parameters('ssd_300_vgg16_ball_v2.params')
            btic = time.time()

def valid_test():
    eval_metric.reset()
    #net.set_nms(nms_thresh=0.45,nms_topk=400)
    net.hybridize()
    for i,batch in enumerate(val_data):
        print('batch %d'%(i+1))
        data=gluon.utils.split_and_load(batch[0],ctx_list=ctx,batch_axis=0,even_split=False)
        label=gluon.utils.split_and_load(batch[1],ctx_list=ctx,batch_axis=0,even_split=False)
        det_bboxes=[]
        det_ids=[]
        det_scores=[]
        gt_bboxes=[]
        gt_ids=[]
        gt_difficults=[]
        for x,y in zip(data,label):
            #get predicton results
            ids,scores,bboxes=net(x)
            print(scores)
            det_ids.append(ids)
            det_scores.append(scores)
            #clip to image size
            det_bboxes.append(bboxes.clip(0,batch[0].shape[2]))
            #split groud truths
            gt_ids.append(y.slice_axis(axis=-1,begin=4,end=5))
            gt_bboxes.append(y.slice_axis(axis=-1, begin=0, end=4))
            gt_difficults.append(y.slice_axis(axis=-1, begin=5, end=6) if y.shape[-1] > 5 else None)
                # update metric
        eval_metric.update(det_bboxes, det_ids, det_scores, gt_bboxes, gt_ids, gt_difficults)
        map_name, mean_ap=eval_metric.get()
        print(map_name,mean_ap)


def test():
    x, image = gcv.data.transforms.presets.ssd.load_test('ball.png',300)
    cid, score, bbox = net(x)

    ax = viz.plot_bbox(image, bbox[0], score[0], cid[0], class_names='ball')
    plt.show() 

if __name__=="__main__":
   train_data,val_data,eval_metric = get_dataloader(net, train_dataset,valid_dataset,300,4,0)
   train()
   #valid_test()
   test()



