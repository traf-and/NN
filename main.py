from NN.module import save_nn
import tkinter as tk
from tkinter import Label, Toplevel, filedialog 
import numpy as np
from numpy.core.fromnumeric import _ndim_dispatcher
from numpy.lib.npyio import loadtxt
from torch.nn import Tanh, Sigmoid, ReLU
import module as md
import script as sc


class Window():
    class Entr_win(tk.Toplevel):
        def __init__(self, num_fld=1, lab_txt=["1"], txt_fld=["1"], title_txt="test", comb_txt=[],comb_lab_txt=[], comb_num=0):
            super().__init__()
            self.str_in=[]
            self.title(title_txt)
            self.minsize(width=400,height=200)
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
                lab_comb[i]=tk.Label(self,width=20, text=comb_lab_txt[i])
                self.comb.append=tk.Combobox(self,values=comb_txt)
                lab_comb.pack()
                self.comb[i].pack()

            but_ac=tk.Button(self, text="Accept", command=self.ins)
            but_ac.pack()
            self.mainloop
        def ins(self):
            for i in self.name:
                self.str_in.append(i.get())
            for i in self.comb:
                self.act=i.current()
            self.destroy()



    class Top(tk.Tk):
        def __init__(self):
            super().__init__()

            self.path = None # file path
            self.nn_in = [] # file IN
            self.nn_out = [] # file OUT
            self.nn_obj = None
            self.pred = [] # preditcion
            self.in_trn = []
            self.in_test=[]
            self.out_trn = []
            self.out_test=[]
            self.title("NN")
            button_open=tk.Button(self,text='Open File',width=25,height=3,font='arial 14', command=self.open_file)
            button_lm=tk.Button(self,text='NN LM',width=25,height=3,font='arial 14', command=self.but_lm)
            button_lin=tk.Button(self,text='NN Lin',width=25,height=3,font='arial 14')
            button_lstm=tk.Button(self,text='NN LSTM',width=25,height=3,font='arial 14')
            button_pred=tk.Button(self,text='Predict',width=25,height=3,font='arial 14')
            button_test=tk.Button(self,text='Test',width=25,height=3,font='arial 14')
            button_save=tk.Button(self,text='Save NN',width=25,height=3,font='arial 14')
            button_load=tk.Button(self,text='Load NN',width=25,height=3,font='arial 14')
            #button_lin=tk.Button(self,text='Text processing',width=25,height=3,font='arial 14')
            button_script=tk.Button(self,text='Script',width=25,height=3,font='arial 14')
            button_close=tk.Button(self,text='Clsoe app',width=25,height=3,font='arial 14', command=self.destroy)

            button_open.pack()
            button_lm.pack()
            button_lin.pack()
            button_lstm.pack()
            button_pred.pack()
            button_test.pack()
            button_save.pack()
            button_load.pack()
            #button_lin.pack()
            button_script.pack()
            button_close.pack()

        def open_file(self):
            self.path  = tk.filedialog.askopenfilename()
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
            #print(a[self.nn_in[0]:self.nn_in[-1], : ])
            self.nn_in=a[self.nn_in[0] : self.nn_in[-1]+1, :]
            self.nn_out=a[self.nn_out[0] : self.nn_out[-1]+1, :]
            self.in_trn, self.in_test=md.crt_valid(self.nn_in)
            self.out_trn, self.out_test=md.crt_valid(self.nn_out)
            
        
        def but_lm(self):
            lay_win=Window.Entr_win(num_fld=1,lab_txt=["Number of hidden layers"], txt_fld=[2], title_txt="Num")
            lab_txt=[]
            txt_fld=[]
            conf=[len(self.in_trn)]
            for i in range(int(lay_win.str_in)):
                lab_txt.append=str(i+1)+"hidden layer"
                txt_fld.append=str(5)
            mod_win=Window.Entr_win(num_fld=int(lay_win.str_in),lab_txt=lab_txt, txt_fld=txt_fld, title_txt="Hidden layers configuration")
            for i in mod_win.str_in:
                conf.append(i)
            conf.append(len(self.out_trn))
            lab_txt=["MSE target","Number of epochs", "Minimum neurons", "Maximum neurons"]
            txt_fld=["0.01","100","3","8"]
            conf_win=Window.Entr_win(num_fld=4,lab_txt=lab_txt, txt_fld=txt_fld, title_txt="Model conguration", comb_txt=["Yes","No"],comb_lab_txt=["Choose the optimal number of neurons"], comb_num=1)
            er_targ=conf_win.str_in[0]
            n_epochs=conf_win.str_in[1]
            min_n=conf_win.str_in[2]
            max_n=conf_win.str_in[3]            
            md.crt_NN(self.nn_obj, self.in_trn, self.out_trn, self.in_test, self.out_test, min_n, max_n, n_epochs, conf, sect_ner, typ="LM")
            #plot y pred, in_train, test
            
        def but_lin(self):
            lay_win=Window.Entr_win(num_fld=1,lab_txt=["Number of hidden layers"], txt_fld=[2], title_txt="Num")
            lab_txt=[]
            txt_fld=[]
            comb_lab_txt=[]
            conf=[]
            tr_funs=[]
            for i in range(int(lay_win.str_in[0])):
                lab_txt.append=str(i+1)+"hidden layer"
                txt_fld.append=str(5)
                if i!=int(lay_win.str_in):
                    comb_lab_txt.append="activation function"
            comb_txt=["Tanh","Sigm","Relu"]
            mod_win=Window.Entr_win(num_fld=int(lay_win.str_in),lab_txt=lab_txt, txt_fld=txt_fld, title_txt="Hidden layers configuration", comb_txt=comb_txt, comb_lab_txt=comb_lab_txt, comb_num=int(lay_win.str_in[0])-1)
            for i in mod_win.str_in:
                conf.append(i)
            for i in mod_win.act:
                if i==0:
                    tr_funs.append(Tanh)
                elif i==1:
                    tr_funs.append(Sigmoid)
                elif i==2:
                    tr_funs.append(ReLU)
            lab_txt=["MSE target","Number of epochs", "Minimum neurons", "Maximum neurons"]
            txt_fld=["0.01","100","3","8"]
            conf_win=Window.Entr_win(num_fld=4,lab_txt=lab_txt, txt_fld=txt_fld, title_txt="Model conguration", comb_txt=["Yes","No"],comb_lab_txt=["Choose the optimal number of neurons"], comb_num=1)
            er_targ=conf_win.str_in[0]
            n_epochs=conf_win.str_in[1]
            min_n=conf_win.str_in[2]
            max_n=conf_win.str_in[3] 

            md.crt_NN(self.nn_obj, self.in_trn, self.out_trn, self.in_test, self.out_test, min_n, max_n, n_epochs, tr_funs, conf, sect_ner, typ="Lin")
            #create plot

        def but_lstm(self):
            lay_win=Window.Entr_win(num_fld=1,lab_txt=["Number of hidden layers"], txt_fld=[2], title_txt="Num")
            lab_txt=[]
            txt_fld=[]
            comb_lab_txt=[]
            conf=[]
            tr_funs=[]
            for i in range(int(lay_win.str_in[0])):
                lab_txt.append=str(i+1)+"hidden layer"
                txt_fld.append=str(5)
                if i!=int(lay_win.str_in):
                    comb_lab_txt.append="activation function"
            comb_txt=["Tanh","Sigm","Relu"]
            mod_win=Window.Entr_win(num_fld=int(lay_win.str_in),lab_txt=lab_txt, txt_fld=txt_fld, title_txt="Hidden layers configuration", comb_txt=comb_txt, comb_lab_txt=comb_lab_txt, comb_num=int(lay_win.str_in[0])-1)
            for i in mod_win.str_in:
                conf.append(i)
            for i in mod_win.act:
                if i==0:
                    tr_funs.append(Tanh)
                elif i==1:
                    tr_funs.append(Sigmoid)
                elif i==2:
                    tr_funs.append(ReLU)
            lab_txt=["MSE target","Number of epochs", "Minimum neurons", "Maximum neurons"]
            txt_fld=["0.01","100","3","8"]
            conf_win=Window.Entr_win(num_fld=4,lab_txt=lab_txt, txt_fld=txt_fld, title_txt="Model conguration", comb_txt=["Yes","No"],comb_lab_txt=["Choose the optimal number of neurons"], comb_num=1)
            er_targ=conf_win.str_in[0]
            n_epochs=conf_win.str_in[1]
            min_n=conf_win.str_in[2]
            max_n=conf_win.str_in[3] 

            md.crt_NN(self.nn_obj, self.in_trn, self.out_trn, self.in_test, self.out_test, min_n, max_n, n_epochs, tr_funs, conf, sect_ner, typ="LSTM")
            #create plot
        
        def but_pred(self):
            out=md.pred(self.nn_obj, self.nn_in)
            a=tk.filedialog.asksaveasfile()
            np.savetxt(a, np.c_[self.nn_in, out], fmt='%1.6f')
        
        def but_test(self):
            test_loss=md.test(self.nn_obj, self.nn_in, self.nn_out)
            loss_win = tk.Toplevel()
            lab = Label(loss_win, text="Mse=" + str(test_loss)+"%")

        def but_save_net(self):
            a=tk.filedialog.asksaveasfile()
            md.save_nn(self.nn_obj, a)

        def but_load_net(self):
            a = tk.filedialog.askopenfilename()
            md.load_nn(a)

        """
        def but_lin(self):
            pass
        """
        def but_script(self):
            pass



root = Window.Top()
root.mainloop()