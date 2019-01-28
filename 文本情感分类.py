import collections,mxnet
import d2lzh as d2l 
from mxnet import gluon,init,nd 
from mxnet.contrib import text 
from mxnet.gluon import data as gdata,loss as gloss,nn,rnn,utils as gutils 
import os 
import random 
import tarfile 


def download_imdb(data_dir='../data'):
    url=('http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz')
    sha1 = '01ada507287d82875905620988597833ad4e0903'
    fname = gutils.download(url, data_dir, sha1_hash=sha1)
    with tarfile.open(fname, 'r') as f:
        print('---------')
        f.extractall(data_dir)

#download_imdb()


def read_imdb(folder='train'):
    data=[]
    for label in ['pos','neg']:
        folder_name=os.path.join('../data/aclImdb/',folder,label)
        for file in os.listdir(folder_name):
            with open(os.path.join(folder_name,file),'rb') as f:
                review=f.read().decode('utf-8').replace('\n','').lower()
                data.append([review,1 if label=='pos' else 0])
    random.shuffle(data)
    return data

train_data,test_data=read_imdb('train'),read_imdb('test')

def get_tokenized_imdb(data):

    def tokenizer(text):
        return [tok.lower() for tok in text.split(' ')]
    
    return [tokenizer(review) for review,_ in data]


def get_vocab_imdb(data):
    tokenized_data=get_tokenized_imdb(data)

    counter=collections.Counter([tk for st in tokenized_data for tk in st])
    return text.vocab.Vocabulary(counter,min_freq=5)

vocab=get_vocab_imdb(train_data)
#print(len(vocab))

def preprocess_imdb(data,vocab):
    max_l=500

    def pad(x):
         return x[:max_l] if len(x)>max_l else x+[0]*(max_l-len(x))


    tokenized_data=get_tokenized_imdb(data)

    features=nd.array([pad(vocab.to_indices(x)) for x in tokenized_data])

    labels =nd.array([score for _,score in data])

    return features,labels


batch_size =64

train_set=gdata.ArrayDataset(*preprocess_imdb(train_data,vocab))

test_set=gdata.ArrayDataset(*preprocess_imdb(test_data,vocab))

train_iter=gdata.DataLoader(train_set,batch_size,shuffle=True)

test_iter=gdata.DataLoader(test_set,batch_size)


#在这个模型中，每个词先通过嵌入层得到特征向量。然后，我们使用双向循环神经网络对特征序列进一步编码得到序列信息。
#最后，我们将编码的序列信息通过全连接层变换为输出。具体来说，我们可以将双向长短期记忆在最初时间步和最终时间步的隐藏状态连接。
#作为序列特征的表征传递给输出层分类，在下面实现BiRNN类中，Embedding实例即嵌入层，LSTM实例即为序列编码的隐藏层，Dense实例即生成
#分类结果的输出层。

class BiRNN(nn.Block):
    def __init__(self,vocab,embed_size,num_hiddens,num_layers,**kwargs):
        super(BiRNN,self).__init__(**kwargs)
        self.embedding=nn.Embedding(len(vocab),embed_size)

        self.encoder=rnn.LSTM(num_hiddens,num_layers=num_layers,
                             bidirectional=True,input_size=embed_size)
        #self.encoder=rnn.GRU(num_hiddens,num_layers=num_layers,input_size=embed_size)
        self.decoder=nn.Dense(2)

    def forward(self,inputs):
        #inputs的形状是(批量大小,词数),因为LSTM需要将序列作为第一维，所以将输入转置后
        #再提取词特征，输出形状为(词数，批量大小,词向量维度)
        embeddings=self.embedding(inputs.T)
        #state形状是(词数,批量大小，28)
        states=self.encoder(embeddings)
        #连接初始时间步和最终时间步的隐藏状态作为全连接输入，它的形状为
        #(批量大小，4*隐藏单元个数)
        encoding=nd.concat(states[0],states[-1])

        outputs=self.decoder(encoding)

        return outputs

embed_size, num_hiddens, num_layers, ctx =100,100, 2, mxnet.cpu()
net = BiRNN(vocab, embed_size, num_hiddens, num_layers)
net.initialize(init.Xavier(), ctx=ctx)

glove_embedding = text.embedding.create(
    'glove', pretrained_file_name='glove.6B.100d.txt', vocabulary=vocab)


net.embedding.weight.set_data(glove_embedding.idx_to_vec)
net.embedding.collect_params().setattr('grad_req', 'null')


lr, num_epochs = 0.01, 5
trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': lr})
loss = gloss.SoftmaxCrossEntropyLoss()
d2l.train(train_iter, test_iter, net, loss, trainer, ctx, num_epochs)
net.save_parameters('sentiment_anaysis')