from logging import root
import numpy as np
import torch as tr
import pyrenn as prn
import tkinter as tk
from tkinter import Canvas, ttk
import numpy as np
from matplotlib.figure import Figure 
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,NavigationToolbar2Tk)
from sys import path, platform
import script as sc



class Entr_win(tk.Toplevel):
    """
    Data entry window class
    """
    def __init__(self, 
    num_fld=1, 
    lab_txt=["1"], 
    txt_fld=["1"], 
    title_txt="test", 
    comb_txt=[],
    comb_lab_txt=[], 
    comb_num=0,  
    root_x=50, 
    root_y=50):
        """
        Create data entry window 
        self: Entr_win 
        num_fld: int, num of forms
        lab_txt: list, form signatures
        txt_fld: list, default text in forms
        comb_txt: list, dropdown values, applies to all dropdowns
        comb_lab_txt: list, dropdown signatures
        comb_num: int, num of dropdown
        """
        super().__init__()
        self.geometry(f'+{root_x}+{root_y}') #head=y+20px
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
            lab[i] = tk.Label(self,width=15, text=lab_txt[i])
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
        """
        Internal function for filling arrays
        """
        for i in self.name:
            self.str_in.append(i.get())
        for i in self.comb:
            self.act.append(i.current())
        self.destroy()

class Continue_Train(tk.Toplevel):
    def __init__(self, err, epochs_sum, root_x=50, root_y=50):
        super().__init__()
        self.title("Continue train?")
        self.answer = False
        lab1 = tk.Label(self, width=20, font='arial 12', text="Current error:").grid(row=0, column=0)
        #lab1.pack()#grid
        lab2 = tk.Label(self, width=20, font='arial 12', text=str(round(err, 4)) + '%').grid(row=1, column=0)
        #lab2.pack()#grid
        lab3 = tk.Label(self, width=20, font='arial 12', text="Total epochs:").grid(row=2, column=0)
        #lab3.pack()#grid
        lab4 = tk.Label(self, width=20, font='arial 12', text=str(epochs_sum)).grid(row=3, column=0)
        #lab4.pack()#grid
        #cont_win.update_idletasks()
        #cont_win.geometry(f'+{root_x}+{root_y}')
        button_yes = tk.Button(self, text='Yes',width=15,height=1, font='arial 14', command=self.yes_command).grid(row=4, column=0)
        button_no = tk.Button(self, text='No',width=15,height=1, font='arial 14', command=self.no_command).grid(row=4, column=1)
        self.mainloop
    
    def yes_command(self):
        self.answer = True
        self.destroy()

    def no_command(self):
        self.answer = False
        self.destroy()

        


class Top(tk.Tk):
    """
    Main window class
    """
    def __init__(self):
        """
        Create main window class 
        """
        super().__init__()
        self.geometry('+50+50')
        #self.root_y = self.winfo_y() self.winfo_width() self.winfo_height()
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
        button_open=tk.Button(self,text='Open File',width=15,height=1,font='arial 14', command=self.but_open_file)
        button_lm=tk.Button(self,text='NN LM',width=15,height=1,font='arial 14', command=self.but_lm)
        button_lin=tk.Button(self,text='NN Lin',width=15,height=1,font='arial 14', command=self.but_lin)
        button_pred=tk.Button(self,text='Predict',width=15,height=1,font='arial 14', command=self.but_pred)
        button_test=tk.Button(self,text='Test',width=15,height=1,font='arial 14', command=self.but_test)
        button_save=tk.Button(self,text='Save NN',width=15,height=1,font='arial 14', command=self.but_save_net)
        button_load=tk.Button(self,text='Load NN',width=15,height=1,font='arial 14', command=self.but_load_net)
        button_script=tk.Button(self,text='Script',width=15,height=1,font='arial 14', command=self.but_script)
        button_close=tk.Button(self,text='Close',width=15,height=1,font='arial 14', command=self.destroy)
        button_open.pack()
        button_lm.pack()
        button_lin.pack()
        button_pred.pack()
        button_test.pack()
        button_save.pack()
        button_load.pack()
        button_script.pack()
        button_close.pack()
        self.update_idletasks()

    def but_open_file(self):
        """ 
        open file and create train and test array
        only work with local variables Top class
        """
        path  = tk.filedialog.askopenfilename(filetypes = [('File','*.txt')])
        if not path:
            return
        f = open(path).readline()
        a=len(f.split())
        if isinstance(self.nn_obj, dict):
            in_end = self.nn_obj['nn'][0]
        elif isinstance(self.nn_obj, Net_tr):
            in_end = self.nn_obj.layers[1].in_features
        else:
            in_end = a-1
        in_win=Entr_win(num_fld=2,lab_txt=["IN start", "IN end"], txt_fld=["1", in_end], title_txt="IN")
        self.wait_window(in_win)
        if not in_win.str_in:
            return
        nn_in=in_win.str_in
        out_win=Entr_win(num_fld=2,lab_txt=["OUT start", "OUT end"], txt_fld=[a, a], title_txt="OUT")
        self.wait_window(out_win)
        if not out_win.str_in:
            return
        nn_out=out_win.str_in
        a=np.loadtxt(path, unpack=True)
        self.nn_in=a[int(nn_in[0])-1: int(nn_in[-1]), :].T
        self.nn_out=a[int(nn_out[0])-1 : int(nn_out[-1]), :].T
        self.path=path

    def but_lm(self):
        """
        create and train pyrenn NN with LM optimization algorithm
        only work with local variables Top class
        """
        if not self.path:
            tk.messagebox.showerror("Error", "Open file first")
            return
        lay_win=Entr_win(num_fld=1,lab_txt=["Number of hidden layers"], txt_fld=[2], title_txt="Num")
        self.wait_window(lay_win)
        if not lay_win.str_in:
            return
        lab_txt=[]
        txt_fld=[]
        conf=[self.nn_in.shape[-1]]
        for i in range(int(lay_win.str_in[0])):
            lab_txt.append(str(i+1)+"hidden layer")
            txt_fld.append(str(20))
        mod_win=Entr_win(num_fld=int(lay_win.str_in[0]),lab_txt=lab_txt, txt_fld=txt_fld, title_txt="Hidden layers configuration")
        self.wait_window(mod_win)
        if not mod_win.str_in:
            return
        for i in mod_win.str_in:
            conf.append(int(i))
        conf.append(self.nn_out.shape[-1])
        lab_txt=["MSE target","Number of epochs", "Minimum neurons", "Maximum neurons", "Train/test coefficient"]
        txt_fld=["0.01","50","15","80", "0.1"]
        conf_win=Entr_win(num_fld=5,lab_txt=lab_txt, txt_fld=txt_fld, title_txt="Model conguration", comb_txt=["No","Yes"],
        comb_lab_txt=["Selection of the number of neurons", "Train/test split","Use retrain?"], comb_num=3)
        self.wait_window(conf_win)
        if not conf_win.str_in:
            return
        er_tar=float(conf_win.str_in[0])
        n_epochs=int(conf_win.str_in[1])
        min_n=int(conf_win.str_in[2])
        max_n=int(conf_win.str_in[3])
        spl_coef=float(conf_win.str_in[4])
        self.withdraw()
        self.nn_obj, conf, vert_coef, hor_coef = lm_NN(self.nn_in.copy(), self.nn_out.copy(), er_tar, self, min_n, max_n, n_epochs, conf=conf, 
        sect_ner=conf_win.act[0],train_test=conf_win.act[1], retrain=conf_win.act[2], spl_coef=spl_coef,
        root_width=self.winfo_width(), root_height=self.winfo_height(), root_x=self.winfo_x(), root_y=self.winfo_y())
        self.deiconify()
        if conf_win.act[1] or conf_win.act[2]:
            x=tr.from_numpy(self.nn_in).float()
            y=tr.from_numpy((self.nn_out)).float()
            dataset = tr.utils.data.TensorDataset(x, y)
            a=int(len(dataset)*(1-spl_coef))
            data_trn,data_test=tr.utils.data.random_split(dataset, [a,int(len(dataset)-a)],generator=tr.Generator().manual_seed(42))
            dataloader = tr.utils.data.DataLoader(data_trn, shuffle=False, batch_size=len(data_trn))
            nn_in_trn=(next(iter(dataloader))[0].numpy())
            nn_out_trn=(next(iter(dataloader))[1].numpy())
            nn_in_test=data_test[:][0].numpy()
            nn_out_test=data_test[:][1].numpy()
            y_trn=pred(self.nn_obj, nn_in_trn).reshape(nn_out_trn.shape)
            y_test=pred(self.nn_obj, nn_in_test).reshape(nn_out_test.shape)
            loss_trn=loss(y_trn, nn_out_trn, self.nn_obj)
            loss_test=loss(y_test, nn_out_test, self.nn_obj)
            self.plot_split(conf, nn_in_trn, nn_out_trn, nn_in_test, nn_out_test, y_trn, y_test, loss_trn, loss_test, vert_coef, hor_coef)
        else:
            y_pred=pred(self.nn_obj, self.nn_in).reshape(self.nn_out.shape)
            pred_loss=loss(y_pred, self.nn_out, self.nn_obj)
            self.plot_orig(conf, self.nn_in, self.nn_out, y_pred, pred_loss, vert_coef)
        
    def but_lin(self):
        """
        create and train pytorch Linear NN with ADAM optimization algorithm
        only work with local variables Top class
        """
        if not self.path:
            tk.messagebox.showerror("Error", "Open file first")
            return
        lab_txt=[]
        txt_fld=[]
        min_ar=[]
        max_ar=[]
        for i in range(self.nn_in.shape[-1]):
            lab_txt.append(str(i+1)+" column minimum")
            txt_fld.append(min(self.nn_in[:,i]))
            lab_txt.append(str(i+1)+" column maximum")
            txt_fld.append(max(self.nn_in[:,i]))
        for i in range(self.nn_out.shape[-1]):
            lab_txt.append(str(i+1+self.nn_in.shape[-1])+" column minimum")
            txt_fld.append(min(self.nn_out[:,i]))
            lab_txt.append(str(i+1+self.nn_in.shape[-1])+" column maximum")
            txt_fld.append(max(self.nn_out[:,i]))
        min_max=Entr_win(num_fld=(self.nn_in.shape[-1]+self.nn_out.shape[-1])*2,lab_txt=lab_txt, txt_fld=txt_fld, title_txt="Num")
        self.wait_window(min_max)
        if not min_max.str_in:
            return
        for i in range(len(min_max.str_in)):
            if i%2==0:
                min_ar.append(float(min_max.str_in[i]))
            else:
                max_ar.append(float(min_max.str_in[i]))
        lay_win=Entr_win(num_fld=1,lab_txt=["Number of hidden layers"], txt_fld=[2], title_txt="Num")
        self.wait_window(lay_win)
        if not lay_win.str_in:
            return
        lab_txt=[]
        txt_fld=[]
        comb_lab_txt=[]
        tr_funs=[]
        for i in range(int(lay_win.str_in[0])):
            lab_txt.append(str(i+1)+" hidden layer")
            txt_fld.append(str(20))
            comb_lab_txt.append(str(i+1)+" layer activation function")
        comb_txt=["Tanh","Sigm","ReLU", "LeakyReLU"]
        mod_win=Entr_win(num_fld=int(lay_win.str_in[0]),lab_txt=lab_txt, txt_fld=txt_fld, title_txt="Hidden layers configuration", 
        comb_txt=comb_txt, comb_lab_txt=comb_lab_txt, comb_num=int(lay_win.str_in[0]))
        self.wait_window(mod_win)
        if not mod_win.str_in:
            return
        conf=[self.nn_in.shape[-1]]
        for i in mod_win.str_in:
            conf.append(int(i))
        conf.append(self.nn_out.shape[-1])
        for i in mod_win.act:
            if i==0:
                tr_funs.append(tr.nn.Tanh())
            elif i==1:
                tr_funs.append(tr.nn.Sigmoid())
            elif i==2:
                tr_funs.append(tr.nn.ReLU())
            elif i==3:
                tr_funs.append(tr.nn.LeakyReLU())
                
        tr_funs.append(0)
        lab_txt=["MSE target","Number of epochs","learning rate", "Minimum neurons", "Maximum neurons", "Train/test coefficient"]
        txt_fld=["0.01","200","0.001","15","80", "0.1"]
        conf_win=Entr_win(num_fld=6,lab_txt=lab_txt, txt_fld=txt_fld, title_txt="Model conguration", comb_txt=["No","Yes"],
        comb_lab_txt=["Choose the optimal number of neurons","Train/test split", "Use retrain?"], comb_num=3)
        self.wait_window(conf_win)
        if not conf_win.str_in:
            return
        er_tar=float(conf_win.str_in[0])
        n_epochs=int(conf_win.str_in[1])
        lr=float(conf_win.str_in[2])
        min_n=int(conf_win.str_in[3])
        max_n=int(conf_win.str_in[4])
        spl_coef=float(conf_win.str_in[5])
        self.withdraw()
        self.nn_obj, conf, vert_coef, hor_coef = torch_NN(self.nn_in.copy(), self.nn_out.copy(), min_ar, max_ar, er_tar, self, min_n, max_n, n_epochs, conf, 
        tr_funs, lr, sect_ner=conf_win.act[0], train_test=conf_win.act[1], retrain=conf_win.act[2], spl_coef=spl_coef,
        root_width=self.winfo_width(), root_height=self.winfo_height(), root_x=self.winfo_x(), root_y=self.winfo_y())
        self.deiconify()
        if conf_win.act[1] or conf_win.act[2]:
            x=tr.from_numpy(self.nn_in).float()
            y=tr.from_numpy((self.nn_out)).float()
            dataset = tr.utils.data.TensorDataset(x, y)
            a=int(len(dataset)*(1-spl_coef))
            data_trn,data_test=tr.utils.data.random_split(dataset, [a,int(len(dataset)-a)],generator=tr.Generator().manual_seed(42))
            dataloader = tr.utils.data.DataLoader(data_trn, shuffle=False, batch_size=len(data_trn))
            nn_in_trn=(next(iter(dataloader))[0].numpy())
            nn_out_trn=(next(iter(dataloader))[1].numpy())
            nn_in_test=data_test[:][0].numpy()
            nn_out_test=data_test[:][1].numpy()
            y_trn=pred(self.nn_obj, nn_in_trn)
            y_test=pred(self.nn_obj, nn_in_test)
            loss_trn=loss(y_trn, nn_out_trn, self.nn_obj).item()
            loss_test=loss(y_test, nn_out_test, self.nn_obj).item()
            self.plot_split(conf, nn_in_trn, nn_out_trn, nn_in_test, nn_out_test, y_trn, y_test, loss_trn, loss_test, vert_coef, hor_coef)
        else:
            y_pred=pred(self.nn_obj, self.nn_in)
            pred_loss=loss(y_pred, self.nn_out, self.nn_obj).item()
            self.plot_orig(conf, self.nn_in, self.nn_out, y_pred, pred_loss, vert_coef)


    def but_pred(self):
        """
        carries out a forecast based on an open NN and saves as a result to a selected file
        only work with local variables Top class
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
        if platform == "linux" or platform == "linux2":
            path=tk.filedialog.asksaveasfilename(filetypes = [('Prediction file','.txt')])
        elif platform == "win32":
            path=tk.filedialog.asksaveasfilename(filetypes = [('Prediction file','.txt')], defaultextension="*.*")
        else:
            path=tk.filedialog.asksaveasfilename(filetypes = [('Prediction file','.txt')])
        np.savetxt(path, np.c_[np.array(self.nn_in), out], fmt='%1.3f')
    
    def but_test(self):
        """
        Calculates loss and display it
        only work with local variables Top class
        """
        if not self.path:
            tk.messagebox.showerror("Error", "Open file first")
            return
        elif not self.nn_obj:
            tk.messagebox.showerror("Error", "Create or open NN")
            return
        elif not self.nn_obj and not self.nn_in:
            tk.messagebox.showerror("Error", "Open file and create NN")
            return
        if isinstance(self.nn_obj, dict):
            y_pred=pred(self.nn_obj, self.nn_in).reshape(self.nn_out.shape)
            test_loss=loss(y_pred, self.nn_out, self.nn_obj)    
        elif isinstance(self.nn_obj, Net_tr):
            y_pred=pred(self.nn_obj, self.nn_in)
            test_loss=loss(y_pred, self.nn_out, self.nn_obj).item()
        tk.messagebox.showinfo("Loss", "Loss=" + str(test_loss)+" %")

    def but_save_net(self):
        """
        Save NN object in file
        only work with local variables Top class
        """
        if isinstance(self.nn_obj, dict):
            if platform == "linux" or platform == "linux2":
                path=tk.filedialog.asksaveasfilename(filetypes = [('LM NN file','.csv')])
            elif platform == "win32":
                path=tk.filedialog.asksaveasfilename(filetypes = [('LM NN file','.csv')], defaultextension="*.*")
            else:
                path=tk.filedialog.asksaveasfilename(filetypes = [('LM NN file','.csv')])
        elif isinstance(self.nn_obj, Net_tr):
            if platform == "linux" or platform == "linux2":
                path=tk.filedialog.asksaveasfilename(filetypes = [("Torch NN file",".pt")])
            elif platform == "win32":
                path=tk.filedialog.asksaveasfilename(filetypes = [("Torch NN file",".pt")], defaultextension="*.*")
            else:
                path=tk.filedialog.asksaveasfilename(filetypes = [("Torch NN file",".pt")])
        else:
            tk.messagebox.showerror("Error", "Crete NN")
            return
        save_nn(self.nn_obj, path)

    def but_load_net(self):
        """
        Load NN object from file
        only work with local variables Top class
        """
        a = tk.filedialog.askopenfilename(filetypes = [('NN file',['*.csv','*.pt'])])
        self.nn_obj=load_nn(a)

    def but_script(self):
        """
        executes a script from a file script.py
        only work with local variables Top class
        """
        sc.sc(self)

    def plot_orig(self, conf, nn_in_trn, nn_out_trn, y_pred, a, vert_coef):
        """
        draw plot with train, prediction data
        show NN congiguration and MSE for test data
        self: Top window class
        conf: list, NN configuration
        nn_in_trn: list, original input
        nn_out_trn: list, original output
        y_pred: list, prediction data
        a: float, loss
        """
        plot_win=tk.Toplevel()
        plot_win.title("Predict data plot")
        fig = Figure(figsize=(5, 4), dpi=80)
        ax=fig.add_subplot(111)
        ax.plot(nn_in_trn[:, 0], nn_out_trn[:, 0], label="Train data")
        ax.plot(nn_in_trn[:, 0], y_pred[:, 0], label="Prediction data")
        ax.set_xlabel('IN[1]')
        ax.set_ylabel('OUT[1]')
        fig.legend()
        canvas = FigureCanvasTkAgg(fig, master=plot_win)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        toolbar = NavigationToolbar2Tk(canvas, plot_win)
        toolbar.update()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        lab1=tk.Label(plot_win,width=35, text="Model configuration"+str(conf))
        lab1.pack()
        lab2=tk.Label(plot_win,width=35, text="Loss= " + str(round(a,4))+ " %")
        lab2.pack()
        plot_win.geometry(f'+{self.winfo_x()+self.winfo_width()+20}+{self.winfo_y() + vert_coef}')

    def plot_split(self, conf, nn_in_trn, nn_out_trn, nn_in_test, nn_out_test, y_trn, y_test, a, b, vert_coef, hor_coef):
        """
          draw plot with train, prediction data
        show NN congiguration and MSE for test data
        self: Top window class
        conf: list, NN configuration
        nn_in_trn: list, original train input
        nn_out_trn: list, original train output
        nn_in_test: list, original test input
        nn_out_test: list, original test output
        y_trn: list, prediction on train data
        y_test: list, prediction on test data
        a: float, loss on train data
        b: float, loss on test data
        """
        plot_win_trn=tk.Toplevel()
        plot_win_trn.title("Train data dot plot")
        fig = Figure(figsize=(5, 4), dpi=100)
        ax=fig.add_subplot(111)
        ax.scatter(nn_in_trn[:, 0], nn_out_trn[:, 0], label="Train data", s=3)
        ax.scatter(nn_in_trn[:, 0], y_trn[:, 0], label="Prediction on train data", s=5)
        ax.set_xlabel('IN[1]')
        ax.set_ylabel('OUT[1]')
        fig.legend()
        canvas = FigureCanvasTkAgg(fig, master=plot_win_trn)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        toolbar = NavigationToolbar2Tk(canvas, plot_win_trn)
        toolbar.update()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        lab1=tk.Label(plot_win_trn,width=35, text="Model configuration"+str(conf))
        lab1.pack()
        lab2=tk.Label(plot_win_trn,width=35, text="Train data loss= "+str(round(a,4))+ " %")
        lab2.pack()
        plot_win_trn.geometry(f'+{self.winfo_x()+self.winfo_width()+20}+{self.winfo_y() + vert_coef}')
        plot_win_trn.update_idletasks()

        plot_win_test=tk.Toplevel()
        plot_win_test.title("Test data dot plot")
        fig = Figure(figsize=(5, 4), dpi=100)
        ax=fig.add_subplot(111)
        ax.scatter(nn_in_test[:, 0], nn_out_test[:, 0], label="Test data", s=3)
        ax.scatter(nn_in_test[:, 0], y_test[:, 0], label="Prediction on test data",s=5)
        ax.set_xlabel('IN[1]')
        ax.set_ylabel('OUT[1]')
        fig.legend()
        canvas = FigureCanvasTkAgg(fig, master=plot_win_test)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        toolbar = NavigationToolbar2Tk(canvas, plot_win_test)
        toolbar.update()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        lab1=tk.Label(plot_win_test,width=35, text="Model configuration"+str(conf))
        lab1.pack()
        lab2=tk.Label(plot_win_test,width=35, text="Test data loss= "+str(round(b,4))+ " %")
        lab2.pack()
        plot_win_test.geometry(f'+{self.winfo_x()+hor_coef+20}+{self.winfo_y() + vert_coef}')
        plot_win_test.update_idletasks()
        

class Net_tr(tr.nn.Module):
    """
    pytorch NN class
    """
    def __init__(self, sizes=[1,1], funs=[tr.nn.Sigmoid()]):
        """
        create torch NN
        sizes: list, model configuration, sizes[0]-num of inputs, sizes[-1]-num of inputs, 
            other- num of neurons on hidden layers (one value=one layer)
        funs: list, list of activation functions
        """
        super().__init__()
        self.alpha = tr.nn.Parameter(data=None, requires_grad=False)
        self.beta = tr.nn.Parameter(data=None, requires_grad=False)
        if isinstance(sizes,(tuple,list)) and isinstance(funs,(tuple,list)):
            assert len(sizes)-1 == len(funs), 'len(funs) должно быть = len(sizes)-1'
        sizes = [1,int(sizes),1] if isinstance(sizes,int) else sizes 
        funs = [funs]*(len(sizes)-1) if not isinstance(funs,(tuple,list)) else funs
        left = sizes[0]
        layers = [tr.nn.BatchNorm1d(left)]
        for right,f in zip(sizes[1:],funs):
            layers.append(tr.nn.Linear(left,right))
            if f:
                layers.append(tr.nn.BatchNorm1d(right))
                layers.append(f)
            left = right
        self.layers = tr.nn.Sequential(*layers)

    def forward(self, x):
        """
        prediction based on input x
        x: tensor, IN data

        return
        predicted data
        """
        in_num=self.layers[1].in_features
        out_num=self.layers[-1].out_features
        x_new=x.detach().clone()
        for i in range(in_num):
            x_new[:,i]=(x[:,i] - self.beta[i]) * self.alpha[i]      
        
        y=self.layers(x_new)
        for i in range(out_num):
            y[:,i]=(y[:,i] / self.alpha[in_num+i]) + self.beta[in_num+i]
        return y

    def alph_bet(self, x, y, min_ar, max_ar):
        """
        computes and guards variables to normalize
        x: tensor, input data
        y: tensor, output data
        min_ar: list, list of minimal for each col
        max: list, list of maximum for each col
        """
        alpha=[]
        beta=[]
        for i in range(x.shape[-1]):
            alpha.append(1 / (max_ar[i] - min_ar[i]))
            beta.append(min_ar[i])
        for i in range(y.shape[-1]):
            alpha.append(1 / (max_ar[i+x.shape[-1]] - min_ar[i+x.shape[-1]]))
            beta.append(min_ar[i+x.shape[-1]])
        self.alpha.data=tr.tensor(alpha)
        self.beta.data=tr.tensor(beta)

def loss(pred, target, nn_obj):
    """
    calculate loss
    pred: list, prediction data
    target: list, original output
    nn_obj: NN object
    return loss
    """
    if isinstance(nn_obj, dict):
        N=pred.shape[0]
        ls=((1/N)*(np.sum(abs(target-pred)/target)))*100
        return ls
    elif isinstance(nn_obj, Net_tr):
        pred=tr.Tensor(pred)
        target=tr.Tensor(target)
        N=pred.size()[0]     
        ls=((1/N)*(tr.sum(abs(target-pred)/target)))*100
        return ls
    

def lm_NN(nn_in, 
          nn_out, 
          er_tar,
          main_win: tk.Toplevel,  
          min_n=0, 
          max_n=0, 
          n_epochs=50, 
          conf=[1,1,1], 
          sect_ner=False, 
          train_test=False, 
          retrain=False, 
          spl_coef=0.1,
          root_width=200, 
          root_height=200, 
          root_x=50, 
          root_y=50):
    """
    create and train pyrenn NN with LM optimization algorithm
    nn_in: list, train NN IN data
    nn_out: list, train OUT data
    er_tar: float, MSE target
    min_n: int, minimum neurons for selection of the number of neurons
    max_n: int, maximum neurons for selection of the number of neurons
    n_epochs: int, maximum NN train epochs
    conf: list, NN configuration, first element- numbers of input, last element - numbers of outputs, other elemnts - number of neuronus on hidden layers
    sect_ner: bool or 0/1, whether to select the number of neurons, True if yes, False if no
    train_test: bool or 0/1, should the input sample be separated into training and test
    retrain: bool or 0/1, overfitting protection

    return
    nn_obj, NN object
    conf: list, NN neurons configuration
    """
    if train_test or retrain:
        x=tr.from_numpy(nn_in).float()
        y=tr.from_numpy((nn_out)).float()
        dataset = tr.utils.data.TensorDataset(x, y)
        a=int(len(dataset)*(1-spl_coef))
        data_trn,data_test=tr.utils.data.random_split(dataset, [a,int(len(dataset)-a)],generator=tr.Generator().manual_seed(42))
        dataloader = tr.utils.data.DataLoader(data_trn, shuffle=False, batch_size=len(data_trn))
        nn_in=(next(iter(dataloader))[0].numpy()).T
        nn_out=(next(iter(dataloader))[1].numpy()).T
        nn_in_test=data_test[:][0].numpy().T
        nn_out_test=data_test[:][1].numpy().T
    else:
        nn_in_test = 0
        nn_out_test = 0
        nn_in=nn_in.T
        nn_out=nn_out.T

    if sect_ner:
        if min_n>max_n:
            print("Error: minimum neurons>maximum neurons")
        for i in range(len(conf)-2):
            conf[i+1] = min_n
        for i in range(1, len(conf)-1):
            min_loss=20000
            b=conf[i]
            for j in range(min_n, max_n+1):
                conf[i] = j
                nn_obj = prn.CreateNN(conf)
                nn_obj = prn.train_LM(nn_in,nn_out,nn_obj,verbose=False,k_max=1,E_stop=1e-10)
                y_pred = prn.NNOut(nn_in, nn_obj)
                a=loss(y_pred, nn_out, nn_obj)
                print("Current configuration:", conf, ";\t Loss:", a, "%")
                if a<min_loss:
                    min_loss=a
                    b=j
            conf[i]=b
        neurons_number_info(conf, root_height + root_y, root_x)
        print("Best configuration:", conf)
        
    nn_obj = prn.CreateNN(conf)
    loss_val=[]
    loss_test=[]
    epochs_sum = 0
    train = True
    lin_tr = None
    can_tr = None
    lin_test = None 
    can_test = None
    vert_coef = None
    hor_coef = None
    while train == True:
        nn_obj, vert_coef, hor_coef, epochs_sum, err, lin_tr, can_tr, lin_test, can_test = train_body_LM(nn_obj, nn_in, nn_in_test, nn_out, nn_out_test, n_epochs, epochs_sum, loss_val, loss_test, er_tar, train_test, retrain, root_width, root_height, root_x, root_y, lin_tr, can_tr, lin_test, can_test, vert_coef, hor_coef)
        train_win = Continue_Train(err, epochs_sum)
        main_win.wait_window(train_win)
        train = train_win.answer
        if not train_win.answer:
            return nn_obj, conf, vert_coef, hor_coef
        #train = False
    return nn_obj, conf, vert_coef, hor_coef

def torch_NN(nn_in, 
             nn_out, 
             min_ar, 
             max_ar, 
             er_tar,
             main_win: tk.Toplevel,  
             min_n=0, 
             max_n=0, 
             n_epochs=200, 
             conf=[1,1,1], 
             funs=[tr.nn.Sigmoid(), 0], 
             lr=0.001, 
             sect_ner=False, 
             train_test=False, 
             retrain=False, 
             spl_coef=0.1,
             root_width=200, 
             root_height=200, 
             root_x=50, 
             root_y=50):
    """
    create and train pytorch NN with ADAM optimization algorithm
    nn_in: list, train NN IN data
    nn_out: list, train OUT data
    min_ar: list, list of minimum for each col
    min_ar: list, list of maximum for each col
    er_tar: float, MSE target
    min_n: int, minimum neurons for selection of the number of neurons
    max_n: int, maximum neurons for selection of the number of neurons
    n_epochs: int, maximum NN train epochs
    conf: list, NN configuration, first element- numbers of input, last element - numbers of outputs, other elemnts - number of neuronus on hidden layers
    funs: lsit, list of activation functions for each layer
    lr: float, learning rate
    sect_ner: bool, whether to select the number of neurons, True if yes, Flase if no
    train_test: bool or 0/1, should the input sample be separated into training and test
    retrain: bool or 0/1, overfitting protection

    return
    nn_obj, NN object
    conf: list, NN neurons configuration
    """
    tr.manual_seed(42)
    in_orig=tr.from_numpy(np.array(nn_in.copy())).float()
    out_orig=tr.from_numpy(np.array(nn_out.copy())).float()
    if train_test or retrain:
        nn_in=tr.from_numpy(np.array(nn_in)).float()
        nn_out=tr.from_numpy(np.array(nn_out)).float()
        dataset = tr.utils.data.TensorDataset(nn_in, nn_out)
        a=int(len(dataset)*(1-spl_coef))
        data_trn, data_test=tr.utils.data.random_split(dataset, [a,int(len(dataset)-a)],generator=tr.Generator().manual_seed(42))
        dataloader = tr.utils.data.DataLoader(data_trn, shuffle=False, batch_size=len(data_trn))
    else:
        nn_in=tr.from_numpy(np.array(nn_in)).float()
        nn_out=tr.from_numpy(np.array(nn_out)).float()
        dataset = tr.utils.data.TensorDataset(nn_in, nn_out)
        dataloader = tr.utils.data.DataLoader(dataset, shuffle=False, batch_size=len(dataset))
        data_test = 0
    if sect_ner:
        for i in range(len(conf)-2):
            conf[i+1] = min_n
        for i in range(1, len(conf)-1):
            min_loss=200000
            b=conf[i]
            for j in range(min_n, max_n+1):
                conf[i]=j
                nn_obj = Net_tr(conf, funs)
                nn_obj.alph_bet(in_orig,out_orig, min_ar, max_ar)
                optimizer = tr.optim.Adam(nn_obj.parameters(), lr=lr)
                y_pred = nn_obj.forward(nn_in)
                loss_val = loss(y_pred, nn_out, nn_obj)
                a=loss_val.item()
                print("Current configuration:", conf, ";\t Loss:", a, "%")
                loss_val.backward()
                optimizer.step()
                if a<min_loss:
                    min_loss=a
                    b=j
            conf[i]=b
        neurons_number_info(conf, root_height + root_y, root_x)
        print("Best configuration:", conf)
    nn_obj = Net_tr(conf, funs)
    optimizer = tr.optim.Adam(nn_obj.parameters(), lr=lr)
    nn_obj.alph_bet(in_orig,out_orig, min_ar, max_ar)
    loss_test=[]
    loss_tr=[]
    epochs_sum = 0
    train = True
    lin_tr = None
    can_tr = None
    lin_test = None 
    can_test = None
    vert_coef = None
    hor_coef = None
    while train == True:
        nn_obj, vert_coef, hor_coef, epochs_sum, err, lin_tr, can_tr, lin_test, can_test = train_body_torch(nn_obj, dataloader, data_test, n_epochs, epochs_sum, optimizer, loss_tr, loss_test, er_tar, train_test, retrain, root_width, root_height, root_x, root_y, lin_tr, can_tr, lin_test, can_test, vert_coef, hor_coef)
        train_win = Continue_Train(err, epochs_sum)
        main_win.wait_window(train_win)
        train = train_win.answer
        if not train_win.answer:
            return nn_obj, conf, vert_coef, hor_coef
        #train = False
    nn_obj.eval()
    return nn_obj, conf, vert_coef, hor_coef

def train_body_LM(nn_obj,
                  nn_in,
                  nn_in_test,
                  nn_out,
                  nn_out_test,
                  n_epochs,
                  epochs_sum,
                  loss_val,
                  loss_test,
                  er_tar,
                  train_test,
                  retrain,
                  root_width,
                  root_height,
                  root_x,
                  root_y,
                  lin_tr, 
                  can_tr,
                  lin_test=None,
                  can_test=None,
                  vert_coef=None,
                  hor_coef=None):
    for i in range(n_epochs):
        nn_obj = prn.train_LM(nn_in,nn_out,nn_obj,verbose=False,k_max=1,E_stop=1e-10)
        y_pred = prn.NNOut(nn_in, nn_obj)
        loss_val.append(loss(y_pred, nn_out, nn_obj))
        print("Train data loss:", loss_val[-1], "%")
        print(i)
        if i == 0 and epochs_sum==0:
            lin_tr, can_tr, vert_coef, hor_coef = loss_plot(i, loss_val, "Train data loss", root_width, root_height, root_x, root_y)
        else:
            update_plot(lin_tr, can_tr, epochs_sum + i, loss_val)
        if train_test or retrain:
            y_pred_test=prn.NNOut(nn_in_test, nn_obj)
            loss_test.append(loss(y_pred_test, nn_out_test, nn_obj))
            print("Test data loss:", loss_test[-1], "%")
            if i == 0 and epochs_sum==0:
                lin_test, can_test, vert_coef, _ = loss_plot(i, loss_test, "Test data loss", hor_coef, root_height, root_x, root_y)
            else:
                update_plot(lin_test, can_test, epochs_sum + i, loss_test)
        if i>3 and retrain:
            if loss_test[-1]*3-(loss_test[-2]+loss_test[-3]+loss_test[-4])>0:
                print("Retraining")
                epochs_sum += i + 1
                return nn_obj, vert_coef, hor_coef, epochs_sum, loss_val[-1], lin_tr, can_tr, lin_test, can_test
        if loss_val[-1]<=er_tar:
            break
    epochs_sum += i + 1
    return nn_obj, vert_coef, hor_coef, epochs_sum, loss_val[-1], lin_tr, can_tr, lin_test, can_test


def train_body_torch(nn_obj,
                     dataloader,
                     data_test,
                     n_epochs,
                     epochs_sum,
                     optimizer,
                     loss_tr,
                     loss_test,
                     er_tar,
                     train_test,
                     retrain,
                     root_width,
                     root_height,
                     root_x,
                     root_y,
                     lin_tr, 
                     can_tr,
                     lin_test=None,
                     can_test=None,
                     vert_coef=None,
                     hor_coef=None):
    for epoch_index in range(n_epochs):
        for nn_in, nn_out in dataloader:
            nn_obj.train()
            optimizer.zero_grad()
            y_pred = nn_obj.forward(nn_in)
            loss_val = loss(y_pred, nn_out, nn_obj)
            loss_tr.append(loss_val.item())
            if epoch_index%10==0:
                print("Train data loss:", loss_val.item(), "%")
            if epoch_index == 0 and epochs_sum==0:
                lin_tr, can_tr, vert_coef, hor_coef = loss_plot(epoch_index, loss_tr, "Train data loss", root_width, root_height, root_x, root_y)
            else:
                update_plot(lin_tr, can_tr, epochs_sum + epoch_index, loss_tr)
            if train_test or retrain:    
                nn_obj.eval()
                y_pred_test=nn_obj.forward(data_test[:][0])
                loss_test.append(loss(y_pred_test, data_test[:][1], nn_obj).item())
                if epoch_index%10==0:
                    print("Test data loss:", loss_test[-1], "%")
                if epoch_index == 0 and epochs_sum==0:
                    lin_test, can_test, vert_coef, _ = loss_plot(epoch_index, loss_test, "Test data loss", hor_coef, root_height, root_x, root_y)
                else:
                    update_plot(lin_test, can_test, epochs_sum + epoch_index, loss_test)
            if loss_val<=er_tar:
                break
            if epoch_index>3 and retrain:
                if loss_test[-1]*3-(loss_test[-2]+loss_test[-3]+loss_test[-4])>0:
                    print("\nRetraining")
                    nn_obj.eval()
                    epochs_sum += epoch_index + 1
                    return nn_obj, vert_coef, hor_coef, epochs_sum, loss_tr[-1], lin_tr, can_tr
            """if epoch_index==n_epochs+11:
                nn_obj.eval()
                epochs_sum += epoch_index
                return nn_obj, vert_coef, hor_coef, epochs_sum, loss_tr[-1]"""
            loss_val.backward()
            optimizer.step()
    epochs_sum += epoch_index + 1
    return nn_obj, vert_coef, hor_coef, epochs_sum, loss_tr[-1], lin_tr, can_tr, lin_test, can_test

def pred(nn_obj, nn_in):
    """ 
    crete prediction with NN and IN data
    nn_obj, NN object
    nn_in: list, IN data

    return 
    y_pred: np.array, predicted data
    """
    if isinstance(nn_obj, dict):
        y_pred = prn.NNOut(np.array(nn_in.T), nn_obj)
    elif isinstance(nn_obj, Net_tr):
        y_pred = nn_obj.forward(tr.from_numpy(np.array(nn_in)).float()).detach().numpy()
    return y_pred

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
        print("canceled")

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
        print("canceled")
        return
    return nn_obj

def loss_plot(x,
              y,
              title,
              root_width,
              root_height,
              root_x=50,
              root_y=50):
    pl_win=tk.Toplevel()
    pl_win.title(title)
    fig = Figure(figsize=(5, 4), dpi=100)
    ax=fig.add_subplot(111)
    ax.set_xlabel('epoch')
    ax.set_ylabel('error, %')
    ax.plot(np.arange(1, x+2), y, 'r')
    canvas = FigureCanvasTkAgg(fig, master=pl_win)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    toolbar = NavigationToolbar2Tk(canvas, pl_win)
    toolbar.update()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
    pl_win.geometry(f'+{root_x+root_width+20}+{root_y}')
    pl_win.update_idletasks()
    vert_coef = pl_win.winfo_y() + pl_win.winfo_height()
    hor_coef = pl_win.winfo_x() + pl_win.winfo_width()
    
    return ax, canvas, vert_coef, hor_coef

def update_plot(ax,canvas,x,y):
    if y[-1]*50<y[0]:
        ax.set_ylim(0.0,y[-1]*50)
    ax.plot(np.arange(1,x+2), y, 'r')    
    canvas.draw()

def neurons_number_info(conf, vert_coef, hor_coef):
    conf_win=tk.Toplevel()
    conf_win.title("Cerrent model configuration")
    lab1=tk.Label(conf_win, width=20, font='arial 12', text="Model configuration:")
    lab1.pack()
    lab2=tk.Label(conf_win, width=20, font='arial 12', text=str(conf))
    lab2.pack()
    conf_win.update_idletasks()
    conf_win.geometry(f'+{hor_coef}+{vert_coef+50}')

