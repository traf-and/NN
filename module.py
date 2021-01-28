import numpy as np
import torch as tr
import pyrenn as prn

class Net_tr(tr.nn.Module):
    def __init__(self, sizes=[1,1], funs=[tr.nn.Sigmoid()], typ="lin"):
        super().__init__()
        if isinstance(sizes,(tuple,list)) and isinstance(funs,(tuple,list)):
            assert len(sizes)-1 == len(funs), 'len(funs) должно быть = len(sizes)-1'
        sizes = [1,int(sizes),1] if isinstance(sizes,int) else sizes 
        funs = [funs]*(len(sizes)-1) if not isinstance(funs,(tuple,list)) else funs
        left = sizes[0]
        layers = []
        for right,f in zip(sizes[1:],funs):
            if typ=="Lin":
                layers.append(tr.nn.Linear(left,right))
            elif typ=="LSTM":
                layers.append(tr.nn.LSTM(left,right))
            if f:
                layers.append(f)
            left = right
        self.layers = tr.nn.Sequential(*layers)
    def forward(self, x):
        return self.layers(x)


def crt_valid(a):
    a=a.tolist()
    a_trn=[]
    a_test=[]
    for i in range(len(a)):
        a_trn.append(a[i][0:int(len(a[i])*0.7)])
        a_trn[i].extend(a[i][int(len(a[i])*0.8):])
        a_test.append(a[i][int(len(a[i])*0.7) : int(len(a[i])*0.8)])
    print(a_trn[1])
    return a_trn, a_test

def loss(pred,target):
    return(((pred - target)**2).mean())

def lm_NN(nn_obj, nn_in, nn_out,  min_n, max_n, n_epochs=50, conf=[1,1,1], sect_ner=True):
    min_loss=20000
    nn_in=np.array(nn_in)
    nn_out=np.array(nn_out)
    if sect_ner:
        for i in range(1, len(conf)-1):
            b=conf[i]
            for j in range(min_n, max_n+1):
                conf[i]=j
                nn_obj = prn.CreateNN(conf)
                nn_obj = prn.train_LM(nn_in,nn_out,nn_obj,verbose=True,k_max=1,E_stop=1e-5)
                y_pred = prn.NNOut(nn_in, nn_obj)
                a=loss(y_pred, nn_out)
                if a<min_loss:
                    min_loss=a
                    b=j
            conf[i]=b
    nn_obj = prn.CreateNN(conf)
    nn_obj = prn.train_LM(nn_in,nn_out,nn_obj,verbose=False,k_max=n_epochs,E_stop=1e-5)
    #TODO: create MSE train graph
    return nn_obj, conf

def torch_NN(nn_obj, nn_in, nn_out,  min_n, max_n, n_epochs=50, conf=[1,1,1], sect_ner=True, typ="Lin"):
    nn_in=tr.from_numpy(np.c_(nn_in)).float()
    nn_out=tr.from_numpy(np.c_(nn_out)).float()
    min_loss=20000
    f_tanh = tr.nn.Tanh()
    f_sigma = tr.nn.Sigmoid()
    f_relu = tr.nn.ReLU()
    if sect_ner:
        for i in range(1, len(conf)-1):
            b=conf[i]
            for j in range(min_n, max_n+1):
                conf[i]=j
                nn_obj = Net_tr(conf,funs=[f_tanh,f_tanh,0])
                y_pred = nn_obj.forward(nn_in)
                a=loss(y_pred, nn_out)
                if a<min_loss:
                    min_loss=a
                    b=j
            conf[i]=b
    nn_obj = Net_tr(sizes=conf, typ=typ, funs=[f_tanh,f_tanh,0],)
    optimizer = tr.optim.Adam(nn_obj.parameters(), lr=0.001)

    for epoch_index in range(n_epochs):
        optimizer.zero_grad()

        y_pred = nn_obj.forward(nn_in)
        loss_val = loss(y_pred, nn_out)

        loss_val.backward()
        
        optimizer.step()
        if epoch_index%500==0:
            print(loss_val)
    return nn_obj, conf


def pred(nn_obj, nn_in):
    if type(nn_obj)=='dict':
        y_pred = prn.NNOut(nn_in, nn_obj)
    elif type(nn_obj)=='__main__.Net':
        y_pred = nn_obj.forward(nn_in)
    return y_pred



def test(nn_obj, nn_in, nn_out):
    if type(nn_obj)=='dict':
        y_pred = prn.NNOut(nn_in, nn_obj)
    elif type(nn_obj)=='__main__.Net':
        y_pred = nn_obj.forward(nn_in)
    return loss(y_pred, nn_out)

def save_nn(nn_obj, path):
    if type(nn_obj)=='dict':
        prn.saveNN(nn_obj, path)
    elif type(nn_obj)=='__main__.Net':
        tr.save(nn_obj, path)
    else:
        print("error")

def load_nn(path):
    if path[-3:-1]=="csv":
        nn_obj = prn.loadNN(path)
    elif path[-2:-1]=="pt":
        nn_obj = tr.load(path)
        nn_obj.eval()
    else:
        print("error")
    return nn_obj

def crt_NN(nn_obj, nn_in, nn_out, nn_in_valid, min_n, max_n, n_epochs, tr_funs, conf=[1,1,1], sect_ner=True, typ="Lin"):
    if typ=="Lin":
        nn_obj,conf=torch_NN(nn_obj, nn_in, nn_out, min_n, max_n, n_epochs, tr_funs, conf, sect_ner, typ="Lin")
        y_pred=nn_obj.forward(nn_in_valid)
    elif typ=="LSTM":
        nn_obj, conf=torch_NN(nn_obj, nn_in, nn_out,  min_n, max_n, n_epochs, tr_funs, conf, sect_ner, typ="LSTM")
        y_pred=nn_obj.forward(nn_in_valid)
    elif typ=="LM":
        nn_obj, conf=lm_NN(nn_obj, nn_in, nn_out,  min_n, max_n, n_epochs, conf, sect_ner)
        y_pred=prn.NNOut(nn_in_valid, nn_obj)
    return nn_obj, y_pred, conf