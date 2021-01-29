import tkinter as tk
from tkinter import Label, Toplevel, filedialog, ttk
import numpy as np
from numpy.lib.npyio import loadtxt
from torch.nn import Tanh, Sigmoid, ReLU
#import matplotlib.pyplot as plt
from matplotlib.figure import Figure 
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,NavigationToolbar2Tk)
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
            self.in_trn, self.in_test=md.crt_valid(self.nn_in)
            self.out_trn, self.out_test=md.crt_valid(self.nn_out)
            
        
        def but_lm(self):
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
            self.nn_obj, self.pred, conf = md.crt_NN(self.nn_obj, self.in_trn, self.out_trn, self.in_test, er_tar, min_n, max_n, n_epochs, conf=conf, sect_ner=conf_win.act[0], typ="LM")
            #__________________
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
            
        def but_lin(self):
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
                if i!=int(lay_win.str_in):
                    comb_lab_txt.append("activation function")
            comb_txt=["Tanh","Sigm","Relu"]
            mod_win=Window.Entr_win(num_fld=int(lay_win.str_in[0]),lab_txt=lab_txt, txt_fld=txt_fld, title_txt="Hidden layers configuration", comb_txt=comb_txt, comb_lab_txt=comb_lab_txt, comb_num=int(lay_win.str_in[0])-1)
            self.wait_window(mod_win)
            if not mod_win.str_in:
                return
            for i in mod_win.str_in:
                conf.append(int(i))
            for i in mod_win.act:
                if i==0:
                    tr_funs.append(Tanh)
                elif i==1:
                    tr_funs.append(Sigmoid)
                elif i==2:
                    tr_funs.append(ReLU)
            lab_txt=["MSE target","Number of epochs", "Minimum neurons", "Maximum neurons"]
            txt_fld=["0.01","20000","3","8"]
            conf_win=Window.Entr_win(num_fld=4,lab_txt=lab_txt, txt_fld=txt_fld, title_txt="Model conguration", comb_txt=["No","Yes"],comb_lab_txt=["Choose the optimal number of neurons"], comb_num=1)
            self.wait_window(conf_win)
            if not conf_win.str_in:
                return
            er_tar=float(conf_win.str_in[0])
            n_epochs=int(conf_win.str_in[1])
            min_n=int(conf_win.str_in[2])
            max_n=int(conf_win.str_in[3])
            self.nn_obj, self.pred, conf = md.crt_NN(self.nn_obj, self.in_trn, self.out_trn, self.in_test, er_tar, min_n, max_n, n_epochs, tr_funs, conf, sect_ner=conf_win.act[0], typ="Lin")
            #create plot

        def but_lstm(self):
            lay_win=Window.Entr_win(num_fld=1,lab_txt=["Number of hidden layers"], txt_fld=[2], title_txt="Num")
            lab_txt=[]
            txt_fld=[]
            comb_lab_txt=[]
            conf=[len(self.in_trn)]
            tr_funs=[]
            for i in range(int(lay_win.str_in[0])):
                lab_txt.append(str(i+1)+"hidden layer")
                txt_fld.append(str(5))
                if i!=int(lay_win.str_in):
                    comb_lab_txt.append("activation function")
            comb_txt=["Tanh","Sigm","Relu"]
            mod_win=Window.Entr_win(num_fld=int(lay_win.str_in[0]),lab_txt=lab_txt, txt_fld=txt_fld, title_txt="Hidden layers configuration", comb_txt=comb_txt, comb_lab_txt=comb_lab_txt, comb_num=int(lay_win.str_in[0])-1)
            for i in mod_win.str_in:
                conf.append(int(i))
            for i in mod_win.act:
                if i==0:
                    tr_funs.append(Tanh)
                elif i==1:
                    tr_funs.append(Sigmoid)
                elif i==2:
                    tr_funs.append(ReLU)
            lab_txt=["MSE target","Number of epochs", "Minimum neurons", "Maximum neurons"]
            txt_fld=["0.01","2000","3","8"]
            conf_win=Window.Entr_win(num_fld=4,lab_txt=lab_txt, txt_fld=txt_fld, title_txt="Model conguration", comb_txt=["No","Yes"],comb_lab_txt=["Choose the optimal number of neurons"], comb_num=1)
            er_tar=float(conf_win.str_in[0])
            n_epochs=int(conf_win.str_in[1])
            min_n=int(conf_win.str_in[2])
            max_n=int(conf_win.str_in[3])

            self.nn_obj, self.pred, conf = md.crt_NN(self.nn_obj, self.in_trn, self.out_trn, self.in_test, er_tar, min_n, max_n, n_epochs, tr_funs, conf, sect_ner=conf_win.act[0], typ="LSTM")
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

        def plot(y):
            plot_win=tk.Toplevel()
            fig = Figure(figsize = (5, 5), dpi = 100)
            plot1 = fig.add_subplot(111)
            plot1.plot(y)
            canvas = FigureCanvasTkAgg(fig, master = plot_win)   
            canvas.draw()
            canvas.get_tk_widget().pack()
            toolbar = NavigationToolbar2Tk(canvas, 
                                   plot_win) 
            toolbar.update() 
            canvas.get_tk_widget().pack()

root = Window.Top()
root.mainloop()