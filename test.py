import tkinter as tk
from tkinter import filedialog 
import numpy as np

class Window(tk.Tk):

    def __init__(self):
        super().__init__()

        self.path = None # file path
        self.str_len = None # columns numbers
        self.nn_in = None # file IN
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
        len(f.split()





root = Window()
root.mainloop()