import tkinter as tk
from tkinter import Toplevel, filedialog 
import numpy as np
from numpy.lib.npyio import loadtxt
import module as md

class Window():
    class Entr_win(tk.Toplevel):
        def __init__(self, num_fld=1, lab_txt=["1"], txt_fld=["1"], title_txt="test",):
            super().__init__()
            self.str_in=[]
            self.title(title_txt)
            self.minsize(width=400,height=200)
            self.name=[0]*num_fld
            ent=[0]*num_fld
            lab=[0]*num_fld
            for i in range(num_fld):
                self.name[i]=tk.StringVar()
                ent[i]=tk.Entry(self,textvariable=self.name[i])
                ent[i].insert(0, txt_fld[i])
                lab[i] = tk.Label(self,width=20)
                lab[i]['text']=lab_txt[i]
                lab[i].pack()
                ent[i].pack()
            but_ac=tk.Button(self, text="Accept", command=self.ins)
            but_ac.pack()
            self.mainloop
        def ins(self):
            for i in self.name:
                self.str_in.append(int(i.get())-1)
            self.destroy()



    class Top(tk.Tk):
        def __init__(self):
            super().__init__()

            self.path = None # file path
            self.nn_in = [] # file IN
            self.nn_out = [] # file OUT
            self.nn_obj = None
            self.pred = []
            self.pred = [] # preditcion
            self.title("NN")
            button_open=tk.Button(self,text='Open File',width=25,height=3,font='arial 14', command=self.open_file)
            button_lm=tk.Button(self,text='NN LM',width=25,height=3,font='arial 14', command=self.but_lm)
            button_adam=tk.Button(self,text='NN ADAM',width=25,height=3,font='arial 14')
            button_lstm=tk.Button(self,text='NN LSTM',width=25,height=3,font='arial 14')
            button_pred=tk.Button(self,text='Predict',width=25,height=3,font='arial 14')
            button_test=tk.Button(self,text='Test',width=25,height=3,font='arial 14')
            #button_lin=tk.Button(self,text='Text processing',width=25,height=3,font='arial 14')
            button_script=tk.Button(self,text='Script',width=25,height=3,font='arial 14')
            button_close=tk.Button(self,text='Clsoe app',width=25,height=3,font='arial 14', command=self.destroy)

            button_open.pack()
            button_lm.pack()
            button_adam.pack()
            button_lstm.pack()
            button_pred.pack()
            button_test.pack()
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
        
        def but_lm(self):
            in_trn, in_test=md.crt_valid(self.nn_in)

            
        def but_adam(self):
            pass
        
        def but_lstm(self):
            pass
        
        def but_pred(self):
            pass
        
        def but_test(self):
            pass
        """
        def but_lin(self):
            pass
        """
        def but_script(self):
            pass



root = Window.Top()
root.mainloop()