import numpy as np
import torch as tr
import pyrenn as prn
import tkinter as tk
from tkinter import Message, ttk
import numpy as np
from numpy.core.records import array
from numpy.lib.npyio import loadtxt
from torch.nn import Tanh, Sigmoid, ReLU
#import matplotlib.pyplot as plt
from matplotlib.figure import Figure 
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,NavigationToolbar2Tk)
import torch as tr
import module as md
import script as sc


class Window():
    class Entr_win(tk.Toplevel):
        def __init__(self, num_fld=1, lab_txt=["1"], txt_fld=["1"], title_txt="test", comb_txt=[],comb_lab_txt=[], comb_num=0):
            super().__init__()
            self.str_in=[]
            self.title(title_txt)

            if comb_txt:
                self.name=[0]*num_fld
                ent=[0]*num_fld
                self.comb=[]
                self.act=[]
                lab=[0]*num_fld
                lab_comb=[0]*comb_num
            else:
                self.name=[0]*num_fld
                ent=[0]*num_fld
                lab=[0]*num_fld
                self.comb=[]
                self.act=[]
            for i in range(num_fld):
                self.name[i]=tk.StringVar()
                ent[i]=tk.Entry(self,textvariable=self.name[i])
                ent[i].insert(0, txt_fld[i])
                lab[i] = tk.Label(self,width=20, text=lab_txt[i])
                lab[i].pack()
                ent[i].pack()
            for i in range(comb_num):
                lab_comb[i]=tk.Label(self,width=35, text=comb_lab_txt[i])
                self.comb.append(ttk.Combobox(self, values=comb_txt))
                lab_comb[i].pack()
                self.comb[i].pack()
                self.comb[i].current(1)

            but_ac=tk.Button(self, text="Accept", command=self.ins)
            but_ac.pack()
            self.mainloop
        def ins(self):
            for i in self.name:
                self.str_in.append(i.get())
            for i in self.comb:
                self.act.append(i.current())
            self.destroy()



    class Top(tk.Tk):
        def __init__(self):
            super().__init__()

            self.path = None # file path
            self.nn_in = [] # file IN
            self.nn_out = [] # file OUT
            self.nn_obj = None # NN object
            self.pred = [] # preditcion
            self.in_trn = [] # train IN list
            self.in_test=[] # train OUT list
            self.out_trn = [] # test IN list
            self.out_test=[] # test OUT list
            self.title("NN")
            button_open=tk.Button(self,text='Open File',width=25,height=3,font='arial 14', command=self.open_file)
            button_lm=tk.Button(self,text='NN LM',width=25,height=3,font='arial 14', command=self.but_lm)
            button_lin=tk.Button(self,text='NN Lin',width=25,height=3,font='arial 14', command=self.but_lin)
            button_pred=tk.Button(self,text='Predict',width=25,height=3,font='arial 14', command=self.but_pred)
            button_test=tk.Button(self,text='Test',width=25,height=3,font='arial 14', command=self.but_test)
            button_save=tk.Button(self,text='Save NN',width=25,height=3,font='arial 14', command=self.but_save_net)
            button_load=tk.Button(self,text='Load NN',width=25,height=3,font='arial 14', command=self.but_load_net)
            #button_lin=tk.Button(self,text='Text processing',width=25,height=3,font='arial 14'. command=self.but_load_net)
            button_script=tk.Button(self,text='Script',width=25,height=3,font='arial 14', command=self.but_script)
            button_close=tk.Button(self,text='Clsoe app',width=25,height=3,font='arial 14', command=self.destroy)

            button_open.pack()
            button_lm.pack()
            button_lin.pack()
            button_pred.pack()
            button_test.pack()
            button_save.pack()
            button_load.pack()
            #button_lin.pack()
            button_script.pack()
            button_close.pack()

        def open_file(self):
            """ 
            open file and create train and test array
            only work with local variables Window.Top class
            """
            self.path = None
            self.path  = tk.filedialog.askopenfilename()
            if not self.path:
                return
            f = open(self.path).readline()
            a=len(f.split())
            in_win=Window.Entr_win(num_fld=2,lab_txt=["IN start", "IN end"], txt_fld=["1", a-1], title_txt="IN")
            self.wait_window(in_win)
            if not in_win.str_in:
                return
            self.nn_in=in_win.str_in
            out_win=Window.Entr_win(num_fld=2,lab_txt=["OUT start", "OUT end"], txt_fld=[a, a], title_txt="OUT")
            self.wait_window(out_win)
            if not out_win.str_in:
                return
            self.nn_out=out_win.str_in
            a=np.loadtxt(self.path, unpack=True)
            self.nn_in=a[int(self.nn_in[0])-1: int(self.nn_in[-1]), :]
            self.nn_out=a[int(self.nn_out[0])-1 : int(self.nn_out[-1]), :]
            self.in_trn, self.in_test=crt_valid(self.nn_in)
            self.out_trn, self.out_test=crt_valid(self.nn_out)
            
        
        def but_lm(self):
            """
            create and train pyrenn NN with LM optimization algorithm
            only work with local variables Window.Top class
            """
            if not self.path:
                tk.messagebox.showerror("Error", "Open file first")
                return
            lay_win=Window.Entr_win(num_fld=1,lab_txt=["Number of hidden layers"], txt_fld=[2], title_txt="Num")
            self.wait_window(lay_win)
            if not lay_win.str_in:
                return
            lab_txt=[]
            txt_fld=[]
            conf=[len(self.in_trn)]
            for i in range(int(lay_win.str_in[0])):
                lab_txt.append(str(i+1)+"hidden layer")
                txt_fld.append(str(5))
            mod_win=Window.Entr_win(num_fld=int(lay_win.str_in[0]),lab_txt=lab_txt, txt_fld=txt_fld, title_txt="Hidden layers configuration")
            self.wait_window(mod_win)
            if not mod_win.str_in:
                return
            for i in mod_win.str_in:
                conf.append(int(i))
            conf.append(len(self.out_trn))
            lab_txt=["MSE target","Number of epochs", "Minimum neurons", "Maximum neurons"]
            txt_fld=["0.01","50","3","8"]
            conf_win=Window.Entr_win(num_fld=4,lab_txt=lab_txt, txt_fld=txt_fld, title_txt="Model conguration", comb_txt=["No","Yes"],comb_lab_txt=["Choose the optimal number of neurons"], comb_num=1)
            self.wait_window(conf_win)
            if not conf_win.str_in:
                return
            er_tar=float(conf_win.str_in[0])
            n_epochs=int(conf_win.str_in[1])
            min_n=int(conf_win.str_in[2])
            max_n=int(conf_win.str_in[3])
            self.nn_obj, conf = crt_NN(self.in_trn, self.out_trn, self.in_test, er_tar, min_n, max_n, n_epochs, conf=conf, sect_ner=conf_win.act[0], typ="LM")
            self.pred = pred(self.nn_obj, self.in_test)
            self.plot(conf)
            
        def but_lin(self):
            """
            create and train pytorch Linear NN with ADAM optimization algorithm
            only work with local variables Window.Top class
            """
            if not self.path:
                tk.messagebox.showerror("Error", "Open file first")
                return
            lay_win=Window.Entr_win(num_fld=1,lab_txt=["Number of hidden layers"], txt_fld=[2], title_txt="Num")
            self.wait_window(lay_win)
            if not lay_win.str_in:
                return
            lab_txt=[]
            txt_fld=[]
            comb_lab_txt=[]
            conf=[len(self.in_trn)]
            tr_funs=[]
            for i in range(int(lay_win.str_in[0])):
                lab_txt.append(str(i+1)+"hidden layer")
                txt_fld.append(str(5))
                comb_lab_txt.append(str(i+1)+"layer activation function")
            comb_txt=["Tanh","Sigm","Relu"]
            mod_win=Window.Entr_win(num_fld=int(lay_win.str_in[0]),lab_txt=lab_txt, txt_fld=txt_fld, title_txt="Hidden layers configuration", comb_txt=comb_txt, comb_lab_txt=comb_lab_txt, comb_num=int(lay_win.str_in[0]))
            self.wait_window(mod_win)
            if not mod_win.str_in:
                return
            for i in mod_win.str_in:
                conf.append(int(i))
            conf.append(len(self.out_trn))
            for i in mod_win.act:
                if i==0:
                    tr_funs.append(Tanh())
                elif i==1:
                    tr_funs.append(Sigmoid())
                elif i==2:
                    tr_funs.append(ReLU())
            tr_funs.append(0)
            lab_txt=["MSE target","Number of epochs", "Minimum neurons", "Maximum neurons", "learning rate"]
            txt_fld=["0.01","20000","3","8", "0.001"]
            conf_win=Window.Entr_win(num_fld=5,lab_txt=lab_txt, txt_fld=txt_fld, title_txt="Model conguration", comb_txt=["No","Yes"],comb_lab_txt=["Choose the optimal number of neurons"], comb_num=1)
            self.wait_window(conf_win)
            if not conf_win.str_in:
                return
            er_tar=float(conf_win.str_in[0])
            n_epochs=int(conf_win.str_in[1])
            min_n=int(conf_win.str_in[2])
            max_n=int(conf_win.str_in[3])
            self.nn_obj, conf = crt_NN(self.in_trn, self.out_trn, self.in_test, er_tar, min_n, max_n, n_epochs, tr_funs, conf, sect_ner=conf_win.act[0], typ="Lin")
            self.pred = pred(self.nn_obj, self.in_test)
            self.plot(conf)

        def but_pred(self):
            """
            carries out a forecast based on an open NN and saves as a result to a selected file
            only work with local variables Window.Top class
            """
            if not self.nn_obj and not self.path:
                tk.messagebox.showerror("Error", "Open file and create NN")
                return
            elif not self.nn_obj:
                tk.messagebox.showerror("Error", "Create NN")
                return
            elif not self.path:
                tk.messagebox.showerror("Error", "Open file first")
                return
            out=pred(self.nn_obj, self.nn_in)
            a=tk.filedialog.asksaveasfile()
            np.savetxt(a, np.c_[np.array(self.nn_in).T, out], fmt='%1.3f')
        
        def but_test(self):
            """
            Calculates MSE and displays it
            only work with local variables Window.Top class
            """
            if not self.path:
                tk.messagebox.showerror("Error", "Open file first")
                return
            elif not self.nn_obj:
                tk.messagebox.showerror("Error", "Create NN")
                return
            elif not self.nn_obj and not self.nn_in:
                tk.messagebox.showerror("Error", "Open file and create NN")
                return
            test_loss=test(self.nn_obj, self.nn_in, self.nn_out)
            tk.messagebox.showinfo("MSE", "Mse=" + str(test_loss)+"%")

        def but_save_net(self):
            """
            Save NN object in file
            only work with local variables Window.Top class
            """
            if isinstance(self.nn_obj, dict):
                a=tk.filedialog.asksaveasfilename(filetypes = [('LM NN file','*.csv')])    
            elif isinstance(self.nn_obj, Net_tr):
                a=tk.filedialog.asksaveasfilename(filetypes = [("Torch NN file","*.pt")])
            else:
                tk.messagebox.showerror("Error", "Crete NN")
                return
            save_nn(self.nn_obj, a)

        def but_load_net(self):
            """
            Load NN object from file
            only work with local variables Window.Top class
            """
            a = tk.filedialog.askopenfilename()
            self.nn_obj=load_nn(a)

        """
        def but_lin(self):
            pass
        """
        def but_script(self):
            """
            executes a script from a file script.py
            only work with local variables Window.Top class
            """
            sc.script(self)

        def plot(self, conf):
            """
            draw plot with train, test, prediction data
            show NN congiguration and MSE for test data
            conf: list, NN configuration
            only work with local variables Window.Top class
            """
            plot_win=tk.Toplevel()
            fig = Figure(figsize=(5, 4), dpi=100)
            fig.add_subplot(111).plot(self.in_trn[0], self.out_trn[0], label="Train data")
            fig.add_subplot(111).plot(self.in_test[0], self.out_test[0], label="Test data")
            fig.add_subplot(111).plot(self.in_test[0], self.pred, label="Prediction data")
            fig.legend()
            

            canvas = FigureCanvasTkAgg(fig, master=plot_win)  # A tk.DrawingArea.
            canvas.draw()
            canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

            toolbar = NavigationToolbar2Tk(canvas, plot_win)
            toolbar.update()
            canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
            lab1=tk.Label(plot_win,width=35, text="Model configuration"+str(conf[1:3]))
            lab1.pack()
            a=loss(pred(self.nn_obj, self.in_test), self.out_test)
            lab2=tk.Label(plot_win,width=35, text="MSE= "+str(round(a,2))+"%")
            lab2.pack()

class Net_tr(tr.nn.Module):
    """
    pytorch NN clacc
    """
    def __init__(self, sizes=[1,1], funs=[tr.nn.Sigmoid()]):
        """
        create torch NN
        """
        super().__init__()
        if isinstance(sizes,(tuple,list)) and isinstance(funs,(tuple,list)):
            assert len(sizes)-1 == len(funs), 'len(funs) должно быть = len(sizes)-1'
        sizes = [1,int(sizes),1] if isinstance(sizes,int) else sizes 
        funs = [funs]*(len(sizes)-1) if not isinstance(funs,(tuple,list)) else funs
        left = sizes[0]
        layers = []
        for right,f in zip(sizes[1:],funs):
            layers.append(tr.nn.Linear(left,right))
            if f:
                layers.append(f)
            left = right
        self.layers = tr.nn.Sequential(*layers)
    def forward(self, x):
        """
        prediction based on input x
        x: list, IN data

        return
        predicted data
        """
        return self.layers(x)


def crt_valid(a):
    """
    create train and test data from a data

    return
    a_trn: list, train data
    a_ test: list, test data
    """
    a=a.tolist()
    a_trn=[]
    a_test=[]
    for i in range(len(a)):
        a_trn.append(a[i][0:int(len(a[i])*0.7)])
        a_trn[i].extend(a[i][int(len(a[i])*0.8):])
        a_test.append(a[i][int(len(a[i])*0.7) : int(len(a[i])*0.8)])
    return a_trn, a_test

def loss(pred,target):
    """
    calculate MSE

    return MSE
    """
    return(((pred - target)**2).mean())

def lm_NN(nn_in, nn_out, er_tar,  min_n, max_n, n_epochs=50, conf=[1,1,1], sect_ner=True):
    """
    create and train pyrenn NN with LM optimization algorithm
    nn_in: list, train NN IN data
    nn_out: list, train OUT data
    er_tar: float, MSE target
    min_n: int, minimum neurons for selection of the number of neurons
    max_n: int, maximum neurons for selection of the number of neurons
    n_epochs: int, maximum NN train epochs
    conf: list, NN configuration, first element- numbers of input, last element - numbers of outputs, other elemnts - number of neuronus on hidden layers
    sect_ner: bool, whether to select the number of neurons, True if yes, Flase if no

    return
    nn_obj, NN object
    conf: list, NN neurons configuration
    """
    nn_in=np.array(nn_in)
    nn_out=np.array(nn_out)
    if sect_ner:
        for i in range(1, len(conf)-1):
            min_loss=20000
            b=conf[i]
            for j in range(min_n, max_n+1):
                conf[i]=j
                nn_obj = prn.CreateNN(conf)
                nn_obj = prn.train_LM(nn_in,nn_out,nn_obj,verbose=False,k_max=1,E_stop=1e-10)
                y_pred = prn.NNOut(nn_in, nn_obj)
                a=loss(y_pred, nn_out)
                if a<min_loss:
                    min_loss=a
                    b=j
            conf[i]=b
            
    nn_obj = prn.CreateNN(conf)
    for i in range(n_epochs):
        nn_obj = prn.train_LM(nn_in,nn_out,nn_obj,verbose=False,k_max=1,E_stop=1e-5)
        y_pred = prn.NNOut(nn_in, nn_obj)
        loss_val = loss(y_pred, nn_out)
        if loss_val<=er_tar:
            break
    print("end")
    return nn_obj, conf

def torch_NN(nn_in, nn_out, er_tar,  min_n, max_n, n_epochs=20000, conf=[1,1,1], funs=[tr.nn.Sigmoid()], sect_ner=True, lr=0.001):
    """
    create and train pytorch NN with ADAM optimization algorithm
    nn_in: list, train NN IN data
    nn_out: list, train OUT data
    er_tar: float, MSE target
    min_n: int, minimum neurons for selection of the number of neurons
    max_n: int, maximum neurons for selection of the number of neurons
    n_epochs: int, maximum NN train epochs
    conf: list, NN configuration, first element- numbers of input, last element - numbers of outputs, other elemnts - number of neuronus on hidden layers
    funs: lsit, list of activation functions for each layer
    sect_ner: bool, whether to select the number of neurons, True if yes, Flase if no

    return
    nn_obj, NN object
    conf: list, NN neurons configuration
    """
    nn_in=tr.from_numpy(np.array(nn_in).T).float()
    nn_out=tr.from_numpy(np.array(nn_out).T).float()
    if sect_ner:
        for i in range(1, len(conf)-1):
            min_loss=200000
            b=conf[i]
            for j in range(min_n, max_n+1):
                conf[i]=j
                nn_obj = Net_tr(conf,funs, typ=typ)
                optimizer = tr.optim.Adam(nn_obj.parameters(), lr=lr)
                y_pred = nn_obj.forward(nn_in)
                loss_val = loss(y_pred, nn_out)
                loss_val.backward()
                optimizer.step()
                a=loss(y_pred, nn_out)
                if a.item()<min_loss:
                    min_loss=a
                    b=j
            conf[i]=b
    nn_obj = Net_tr(sizes=conf, funs=funs, typ=typ)
    optimizer = tr.optim.Adam(nn_obj.parameters(), lr=lr)
    for epoch_index in range(n_epochs):
        optimizer.zero_grad()
        y_pred = nn_obj.forward(nn_in)
        loss_val = loss(y_pred, nn_out)
        if loss_val<=er_tar:
            break
        loss_val.backward()
        optimizer.step()
        if epoch_index%500==0:
            print(loss_val.item())
    return nn_obj, conf


def pred(nn_obj, nn_in):
    """ 
    crete prediction with NN and IN data
    nn_obj, NN
    nn_in: list, IN data

    return 
    y_pred: np.array, predicted data
    """
    if isinstance(nn_obj, dict):
        y_pred = prn.NNOut(np.array(nn_in), nn_obj)
    elif isinstance(nn_obj, Net_tr):
        y_pred = nn_obj.forward(tr.from_numpy(np.array(nn_in).T).float()).detach().numpy()
    return y_pred

def test(nn_obj, nn_in, nn_out):
    """
    calculate MSE for NN
    nn_obj, NN
    nn_in: lsit, IN data
    nn_out: list, OUT data
    
    return
    MSE
    """
    if isinstance(nn_obj, dict):
        y_pred = prn.NNOut(nn_in, nn_obj)
    elif isinstance(nn_obj, Net_tr):
        y_pred = (nn_obj.forward(tr.from_numpy(np.array(nn_in).T).float())).detach().numpy()
    return loss(y_pred, nn_out)

def save_nn(nn_obj, path):
    """
    save NN object as file
    nn_obj, NN
    path: str, PATH to the saved file
    """
    if isinstance(nn_obj, dict):
        prn.saveNN(nn_obj, path)
    elif isinstance(nn_obj, Net_tr):
        tr.save(nn_obj, path)
    else:
        print("error")

def load_nn(path):
    """
    load NN object from file
    path: str, PATH to the loaded file

    return NN object
    """
    
    if path[-3:]=="csv":
        nn_obj = prn.loadNN(path)
    elif path[-2:]=="pt":
        nn_obj = tr.load(path)
        nn_obj.eval()
    else:
        print("error")
        return
    return nn_obj

def crt_NN(nn_in, nn_out, er_tar, min_n, max_n, n_epochs, tr_funs=[], conf=[1,1,1], sect_ner=True, typ="Lin", lr=0.001):
    """
    create and train NN
    nn_in: list, train NN IN data
    nn_out: list, train OUT data
    er_tar: float, MSE target
    min_n: int, minimum neurons for selection of the number of neurons
    max_n: int, maximum neurons for selection of the number of neurons
    n_epochs: int, maximum NN train epochs
    conf: list, NN configuration, ONLY for torch NN, first element- numbers of input, last element - numbers of outputs, other elemnts - number of neuronus on hidden layers
    funs: list, list of activation functions for each layer
    sect_ner: bool, whether to select the number of neurons, True if yes, Flase if no
    typ: str, type of neural network required, 'Lin' for torch FFNN with ADAM optimizer, 'LM' for pyrenn FFNN with LM optimizer
    lr: float, lerning rate, ONLY for torch NN
    
    return
    nn_obj, NN object
    conf: list, NN neurons configuration
    """
    if typ=="Lin":
        nn_obj,conf=torch_NN(nn_in, nn_out, er_tar, min_n, max_n, n_epochs, conf, tr_funs, sect_ner, lr=lr)
    elif typ=="LM":
        nn_obj, conf=lm_NN(nn_in, nn_out, er_tar,  min_n, max_n, n_epochs, conf, sect_ner)
    return nn_obj, conf