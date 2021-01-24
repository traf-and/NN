import tkinter as tk
from tkinter import Toplevel, filedialog 
import numpy as np

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
                self.str_in.append(i.get())
            self.destroy()



    class Top(tk.Tk):
        def __init__(self):
            super().__init__()

            self.path = None # file path
            self.nn_in = [] # file IN
            self.nn_out = None # file OUT
            self.pred = None # preditcion
            self.title("NN")
            button_open=tk.Button(self,text='Open File',width=25,height=3,font='arial 14', command=self.open_file)
            button_lm=tk.Button(self,text='NN LM',width=25,height=3,font='arial 14')
            button_adam=tk.Button(self,text='NN ADAM',width=25,height=3,font='arial 14')
            button_lstm=tk.Button(self,text='NN LSTM',width=25,height=3,font='arial 14')
            button_pred=tk.Button(self,text='Predict',width=25,height=3,font='arial 14')
            #button_lin=tk.Button(self,text='Text processing',width=25,height=3,font='arial 14')
            button_script=tk.Button(self,text='Script',width=25,height=3,font='arial 14')
            button_close=tk.Button(self,text='Clsoe app',width=25,height=3,font='arial 14')

            button_open.pack()
            button_lm.pack()
            button_adam.pack()
            button_lstm.pack()
            button_pred.pack()
            #button_lin.pack()
            button_script.pack()
            button_close.pack()

        def open_file(self):
            path_to_file = tk.filedialog.askopenfilename()
            self.path = path_to_file
            f = open(self.path).readline()
            a=len(f.split())
            in_win=Window.Entr_win(num_fld=2,lab_txt=["IN start", "IN end"], txt_fld=["1", a-1], title_txt="IN")
            self.wait_window(in_win)
            self.nn_in=in_win.str_in
            out_win=Window.Entr_win(num_fld=2,lab_txt=["OUT start", "OUT end"], txt_fld=[a, a], title_txt="OUT")
            self.wait_window(out_win)
            self.nn_out=out_win.str_in
            




root = Window.Top()
root.mainloop()