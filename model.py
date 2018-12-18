#coding utf-8 
from ssd import *
import time
from mxnet import autograd as ag
from mxnet.contrib.ndarray import MultiBoxDetection
import logging
from mxnet.gluon import loss as gloss
head = '%(asctime)-15s %(message)s'
logging.basicConfig(level=logging.DEBUG, format=head)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler("log.txt")
fh.setLevel(logging.DEBUG)
logger.addHandler(fh)
def train(net,start_epoch,epochs,lr,ctx):
    cls_loss = gloss.SoftmaxCrossEntropyLoss()
    box_loss = gloss.L1Loss()
    cls_metric = mx.metric.Accuracy()
    box_metric = mx.metric.MAE()
    data_shape = 256
    batch_size = 1
    train_data, test_data, class_names, num_class = get_iterators(data_shape, batch_size)
    train_data.reshape(label_shape=(6, 5))
    train_data = test_data.sync_label_shape(train_data)
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr, 'wd': 0.0005})
    log_interval = 1
    for epoch in range(start_epoch, epochs+start_epoch):
        # reset iterator and tick

        train_data.reset()
        cls_metric.reset()
        box_metric.reset()
        tic = time.time()
        # iterate through all batch
        for i, batch in enumerate(train_data):
            #print(i)
            btic = time.time()
            # record gradients
            with ag.record():
                x = batch.data[0].as_in_context(ctx)
                y = batch.label[0].as_in_context(ctx)
                default_anchors, class_predictions, box_predictions = net(x)
                box_target, box_mask, cls_target = training_targets(default_anchors, class_predictions, y)
                # losses
                loss1 = cls_loss(class_predictions, cls_target)
                loss2 = box_loss(box_predictions, box_target, box_mask)
                # sum all losses
                loss = 0.1*loss1 + loss2
                # backpropagate
                loss.backward()
            # apply
            trainer.step(batch_size)
            # update metrics
            cls_metric.update([cls_target], [nd.transpose(class_predictions, (0, 2, 1))])
            box_metric.update([box_target], [box_predictions * box_mask])
            if (i + 1) % 10 == 0:
                name1, val1 = cls_metric.get()
                name2, val2 = box_metric.get()
                print('[Epoch %d Batch %d] speed: %f samples/s, training: %s=%f, %s=%f'
                      %(epoch ,i, batch_size/(time.time()-btic), name1, val1, name2, val2))
            if (i+1)%2000 == 0:
                net.save_params('ssd_%d.params'%(i))
        # end of epoch logging
        name1, val1 = cls_metric.get()
        name2, val2 = box_metric.get()
        print('[Epoch %d] training: %s=%f, %s=%f'%(epoch, name1, val1, name2, val2))
        print('[Epoch %d] time cost: %f'%(epoch, time.time()-tic))
        valid(net,test_data,epoch,ctx)
    # we can save the trained parameters to disk
    net.save_params('ssd_%d.params' % (epochs+start_epoch))

def valid(net,valid_data,epoch,ctx):
    cls_metric = mx.metric.Accuracy()
    box_metric = mx.metric.MAE()

    # reset iterator and tick
    valid_data.reset()
    cls_metric.reset()
    box_metric.reset()
    tic = time.time()
    # iterate through all batch
    for i, batch in enumerate(valid_data):
        with ag.record():
            x = batch.data[0].as_in_context(ctx)
            y = batch.label[0].as_in_context(ctx)
            default_anchors, class_predictions, box_predictions = net(x)
            box_target, box_mask, cls_target = training_targets(default_anchors, class_predictions, y)
        # update metrics
        cls_metric.update([cls_target], [nd.transpose(class_predictions, (0, 2, 1))])
        box_metric.update([box_target], [box_predictions * box_mask])
    name1, val1 = cls_metric.get()
    print('[Epoch %d] time cost: %f'%(epoch, time.time()-tic))
def test(net,ctx):#,image):
    #image = cv2.imread('/BELLO/dog/data/normalsize/n02091244_3075.jpg')
    image = cv2.imread('/BELLO/dog/data/dog3.jpg')
    x = preprocess(image)
    anchors, cls_preds, box_preds = net(x.as_in_context(ctx))
    cls_probs = nd.SoftmaxActivation(nd.transpose(cls_preds, (0, 2, 1)), mode='channel')
    output = MultiBoxDetection(*[cls_probs, box_preds, anchors], force_suppress=True, clip=False)
    #print(output[0].asnumpy())
    display(image[:, :, (2, 1, 0)], output[0].asnumpy(), thresh=0.7)
def get_iterators(data_shape, batch_size):
    class_names = ['dog']
    num_class = len(class_names)
    train_iter = image.ImageDetIter(
        batch_size=batch_size,
        data_shape=(3, data_shape, data_shape),
        path_imglist='/BELLO/dog/data/train.lst',
        path_root='/BELLO/dog/data/normalsize/',
        mean=True)
    val_iter = image.ImageDetIter(
        batch_size=batch_size,
        data_shape=(3, data_shape, data_shape),
        path_imglist='/BELLO/dog/data/valid.lst',
        path_root='/BELLO/dog/data/normalsize/',
        mean=True)
    return train_iter, val_iter, class_names, num_class
