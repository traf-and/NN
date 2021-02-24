from operator import le
import numpy as np
import torch as tr
import pyrenn as prn
import module as md

def sc(Top):
    #for i in range(3):
    Top.but_open_file()
    Top.nn_obj, conf=md.torch_NN(nn_in=Top.nn_in, nn_out=Top.nn_out, er_tar=0.01, n_epochs=20000, conf=[2,15,20,1], funs=[tr.nn.Sigmoid(), tr.nn.Sigmoid(), 0], sect_ner=False, lr=0.001)
    Top.pred=md.pred(Top.nn_obj, Top.nn_in)
    a=0
    Top.plot(conf)
"""    print(pred)
    print("_________")
    print(Top.nn_in)"""