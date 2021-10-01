# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 11:59:18 2020

@author: kf4
"""

from tkinter import *
from tkinter import ttk, colorchooser
from tkinter.filedialog import askopenfile

import numpy as np
import scipy.io

import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

from models.networks import *

from PIL import Image
from PIL import ImageGrab

import win32gui

globvar = 0
globfile = None

def Convert1(img): 
    tens = process1(img)
    Tensor = torch.cuda.FloatTensor 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    netS_A = SPADEGenerator(num_class=4)
    if torch.cuda.device_count() >= 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        netS_A = nn.DataParallel(netS_A)
        netS_A.to(device)
        netS_A.load_state_dict(torch.load('saved_models/2-class.pth'))
        torch.set_grad_enabled(False)
        torch.cuda._lazy_init()
        netS_A = netS_A.eval() 
        print("Let's use", torch.cuda.device_count(), "GPUs!")

    z = torch.randn(1, 256,dtype=torch.float32, device=device) *1                                
    fake_ir = netS_A(tens.type(Tensor),z=z)
    transform1 = transforms.ToPILImage()
    img = transform1(fake_ir.data.cpu().squeeze(0)*.5+.5)
    img.save('GUI outputs/Synth.png') 
    
def process1(img):
    img = np.array(img.getdata()).reshape(img.size[0], img.size[1], 3)
    img = img[2:-2,2:-2,:]
    img2 = np.zeros((512,512))
    L = img2*0
    img2[img[:,:,2]==255]=0
    img2[img[:,:,2]==233]=1
    img2[img[:,:,2]==49]=2
    temp = np.zeros((512,512,4))
    for i in range(4):
        L = img2*0
        L[img2==i] = 1
        temp[:,:,i:i+1] = np.expand_dims(L, axis=2)
    trans1 = transforms.ToTensor()
    temp = trans1(temp)
    return temp.unsqueeze(0)
    
def Convert2(img): 
    tens = process2(img)
    Tensor = torch.cuda.FloatTensor 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    netS_A = SPADEGenerator(num_class=4)
    if torch.cuda.device_count() >= 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        netS_A = nn.DataParallel(netS_A)
        netS_A.to(device)
        netS_A.load_state_dict(torch.load('saved_models/3-class.pth'))
        torch.set_grad_enabled(False)
        torch.cuda._lazy_init()
        netS_A = netS_A.eval() 
        print("Let's use", torch.cuda.device_count(), "GPUs!")

    z = torch.randn(1, 256,dtype=torch.float32, device=device) *1                                
    fake_ir = netS_A(tens.type(Tensor),z=z)
    transform1 = transforms.ToPILImage()
    img = transform1(fake_ir.data.cpu().squeeze(0)*.5+.5)
    img.save('GUI outputs/Synth.png') 
    
def process2(img):
    img = np.array(img.getdata()).reshape(img.size[0], img.size[1], 3)
    img = img[2:-2,2:-2,:]
    img2 = np.zeros((512,512))
    L = img2*0
    img2[img[:,:,2]==255]=0
    img2[img[:,:,2]==233]=1
    img2[img[:,:,2]==49]=2
    img2[img[:,:,2]==7]=3
    temp = np.zeros((512,512,4))
    for i in range(4):
        L = img2*0
        L[img2==i] = 1
        temp[:,:,i:i+1] = np.expand_dims(L, axis=2)
    trans1 = transforms.ToTensor()
    temp = trans1(temp)
    return temp.unsqueeze(0)

def Convert3(img): 
    tens = process3(img)

    Tensor = torch.cuda.FloatTensor 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    netS_A = SPADEGenerator(num_class=11)
    if torch.cuda.device_count() >= 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        netS_A = nn.DataParallel(netS_A)
        netS_A.to(device)
        netS_A.load_state_dict(torch.load('saved_models/10-class.pth'))
        torch.set_grad_enabled(False)
        torch.cuda._lazy_init()
        netS_A = netS_A.eval() 
        print("Let's use", torch.cuda.device_count(), "GPUs!")

    z = torch.randn(1, 256,dtype=torch.float32, device=device) *1                                
    fake_ir = netS_A(tens.type(Tensor),z=z)
    transform1 = transforms.ToPILImage()
    img = transform1(fake_ir.data.cpu().squeeze(0)*.5+.5)
    img.save('GUI outputs/Synth.png') 
    
def process3(img):
    img = np.array(img.getdata()).reshape(img.size[1], img.size[0], 3)

    
    img = img[2:-3,2:-2,:]

    img2 = np.zeros((512,512))
    L = img2*0
    img2[img[:,:,2]==85]=1
    img2[img[:,:,2]==173]=0
    img2[img[:,:,2]==178]=2
    img2[img[:,:,2]==68]=3
    img2[img[:,:,2]==30]=4
    img2[img[:,:,2]==46]=5
    img2[img[:,:,2]==134]=6
    img2[img[:,:,2]==153]=9
    img2[img[:,:,2]==216]=8
    img2[img[:,:,2]==138]=7
    img2[img[:,:,2]==255]=10
#    img2[img[:,:,2]==233]=1
#    img2[img[:,:,2]==49]=2
    temp = np.zeros((512,512,11))
    for i in range(11):
        L = img2*0
        L[img2==i] = 1
        temp[:,:,i:i+1] = np.expand_dims(L, axis=2)
    trans1 = transforms.ToTensor()
    temp = trans1(temp)

    return temp.unsqueeze(0)   
def getter(canvas1,canvas2):
    global globfile

    HWND = canvas1.winfo_id()  # get the handle of the canvas

    rect = win32gui.GetWindowRect(HWND)  # get the coordinate of the canvas

    img = ImageGrab.grab(rect).save("GUI outputs/Labels_creation.png")
    img = Image.open("GUI outputs/Labels_creation.png")  
    if globvar==1:
        Convert1(img)
    if globvar==2:
        Convert2(img)
    if globvar==3:
        Convert3(img)

    my_images = PhotoImage(file =r'GUI outputs/Synth.png')
    root.my_images = my_images # to prevent the image garbage collected
    canvas2.create_image((0,0), image=my_images, anchor='nw')
    my_images2 = PhotoImage(file =r'GUI outputs/Labels_creation.png')       
    root.my_images2 = my_images2 # to prevent the image garbage collected
    canvas1.create_image((0,0), image=my_images2, anchor='nw')
def open_file_1(canvas): 
    file = askopenfile(initialdir = "2-class prostate/",mode ='r', filetypes =[('Image Files', '*.png')]) 
    global globfile
    globfile = file.name
    my_images = PhotoImage(file =globfile)       
    root.my_images = my_images # to prevent the image garbage collected
    canvas.create_image((0,0), image=my_images, anchor='nw')

def open_file_2(canvas): 
    file = askopenfile(initialdir = "3-class prostate/",mode ='r', filetypes =[('Image Files', '*.png')]) 
    global globfile
    globfile = file.name
    my_images = PhotoImage(file =globfile)       
    root.my_images = my_images # to prevent the image garbage collected
    canvas.create_image((0,0), image=my_images, anchor='nw')

def open_file_3(canvas): 
    file = askopenfile(initialdir = "10-class colon/",mode ='r', filetypes =[('Image Files', '*.png')]) 
    global globfile
    globfile = file.name
    my_images = PhotoImage(file =globfile)       
    root.my_images = my_images # to prevent the image garbage collected
    canvas.create_image((0,0), image=my_images, anchor='nw')
    
class main:
    def __init__(self,master):
        self.Window1 = Toplevel(root)


        self.master = master
        self.color_fg = '#54c6e9'
        self.color_bg = 'white'
        self.old_x = None
        self.old_y = None
        self.penwidth = 5
        Button(self.Window1, text="2-class prostate model",relief = 'raised',font = 'arial 18', bg= "#53626F", fg = "white",
               width=20,pady = 10,padx = 10, borderwidth=10, command=self.callback1).grid(row=0,column = 0)
        
        Button(self.Window1, text="3-class prostate model",relief = 'raised',font = 'arial 18', bg= "#EB5257", fg = "white",
               width=20,pady = 10,padx = 10, borderwidth=10, command=self.callback2).grid(row=1,column = 0)
        
        Button(self.Window1, text="10-class colon model",relief = 'raised',font = 'arial 18', bg= "#56AEEA,", fg = "white",
               width=20,pady = 10,padx = 10, borderwidth=10, command=self.callback3).grid(row=2,column = 0)
        
        self.Window1.title('Choose your model')
        self.Window1.geometry('+10+10')
    def callback1(self):
        self.change_window()
        global globvar
        globvar = 1
        self.drawWidgets1(globvar)
        self.c.bind('<B1-Motion>',self.paint)#drwaing the line 
        self.c.bind('<ButtonRelease-1>',self.reset)
    def callback2(self):
        self.change_window()
        global globvar
        globvar = 2
        self.drawWidgets2(globvar)
        self.c.bind('<B1-Motion>',self.paint)#drwaing the line 
        self.c.bind('<ButtonRelease-1>',self.reset)
    def callback3(self):
        self.change_window()
        global globvar
        globvar = 3
        self.drawWidgets3(globvar)
        self.c.bind('<B1-Motion>',self.paint)#drwaing the line 
        self.c.bind('<ButtonRelease-1>',self.reset)
    def change_window(self):
        #remove the other window entirely
        self.Window1.destroy()
    
        #make root visible again
        root.iconify()
        root.deiconify()
    def paint(self,e):
        if self.old_x and self.old_y:
            self.c.create_line(self.old_x,self.old_y,e.x,e.y,width=self.penwidth,fill=self.color_fg,capstyle=ROUND,smooth=True)

        self.old_x = e.x
        self.old_y = e.y

    def reset(self,e):    #reseting or cleaning the canvas 
        self.old_x = None
        self.old_y = None      

    def changeW(self,e): #change Width of pen through slider
        self.penwidth = e
           

    def clear(self):
        self.c.delete(ALL)
    def background(self):  #changing the pen color
        self.color_fg="white"
    def epi(self):  #changing the pen color
        self.color_fg="#fcf731"
    def non_epi(self):  #changing the pen color
        self.color_fg="#54c6e9"
    def nuclei(self):  #changing the pen color
        self.color_fg="#6C0307"
    def change_fg(self):  #changing the pen color
        self.color_fg=colorchooser.askcolor(color=self.color_fg)[1]

    def change_bg(self):  #changing the background color canvas
        self.color_bg=colorchooser.askcolor(color=self.color_bg)[1]
        self.c['bg'] = self.color_bg

    def drawWidgets1(self,mode):
        
        self.controls = Frame(self.master,padx = 5,pady = 5)
        Label(self.controls, text='Pen Width:',font=('arial 12')).grid(row=0,column=0)
        self.slider = ttk.Scale(self.controls,from_= 1, to = 200,command=self.changeW,orient=HORIZONTAL,length = 200)
        self.slider.set(self.penwidth)
        self.slider.grid(row=0,column=1)
        self.controls.pack(side=TOP)
        
        self.controls2 = Frame(self.master)
        self.c1=Button(self.controls2,fg = 'black',bg="#54c6e9",relief = "flat",font = 'arial 18',text="non_epi",command = self.non_epi,width=10)
        self.c1.grid(row=0,column = 0)
        self.c1=Button(self.controls2,fg = 'black',bg="white",relief = "flat",font = 'arial 18',text="background",command = self.background,width=10)
        self.c1.grid(row=1,column = 0)
        self.c1=Button(self.controls2,fg = 'black',bg="#fcf731",relief = "flat",font = 'arial 18',text="epi",command = self.epi,width=10)
        self.c1.grid(row=2,column = 0)

        self.controls2.pack(side=LEFT, anchor=NW)
        
        
        self.c = Canvas(self.master,width=512,height=512,bg=self.color_bg,)
        self.c2 = Canvas(self.master,width=512,height=512,bg=self.color_bg,)
        self.c.pack(fill=BOTH,expand=True,side = LEFT)
        self.c2.pack(fill=BOTH,expand=True,side = RIGHT)
        menu = Menu(self.master)
        self.master.config(menu=menu)
        filemenu = Menu(menu)
        colormenu = Menu(menu)
        menu.add_cascade(label='File',menu=colormenu)
        colormenu.add_command(label='Load',command=lambda:open_file_1(self.c))
        optionmenu = Menu(menu)
        menu.add_cascade(label='Options',menu=optionmenu)
        optionmenu.add_command(label='Clear Canvas',command=self.clear)
        optionmenu.add_command(label='Exit',command=self.master.destroy) 
        button4=Button(root,fg="green",text="convert",command=lambda:getter(self.c,self.c2))
        button4.pack(side=RIGHT)
        self.master.geometry('+10+10')
        
    def drawWidgets2(self,mode):
        
        self.controls = Frame(self.master,padx = 5,pady = 5)
        Label(self.controls, text='Pen Width:',font=('arial 12')).grid(row=0,column=0)
        self.slider = ttk.Scale(self.controls,from_= 1, to = 200,command=self.changeW,orient=HORIZONTAL,length = 200)
        self.slider.set(self.penwidth)
        self.slider.grid(row=0,column=1)
        self.controls.pack(side=TOP)
        
        self.controls2 = Frame(self.master)
        self.c1=Button(self.controls2,fg = 'black',bg="#54c6e9",relief = "flat",font = 'arial 18',text="non_epi",command = self.non_epi,width=10)
        self.c1.grid(row=0,column = 0)
        self.c1=Button(self.controls2,fg = 'black',bg="white",relief = "flat",font = 'arial 18',text="background",command = self.background,width=10)
        self.c1.grid(row=1,column = 0)
        self.c1=Button(self.controls2,fg = 'black',bg="#fcf731",relief = "flat",font = 'arial 18',text="epi",command = self.epi,width=10)
        self.c1.grid(row=2,column = 0)
        self.c1=Button(self.controls2,fg = 'black',bg="#6C0307",relief = "flat",font = 'arial 18',text="nuclei",command = self.nuclei,width=10)
        self.c1.grid(row=3,column = 0)
        self.controls2.pack(side=LEFT, anchor=NW)
        
        
        self.c = Canvas(self.master,width=512,height=512,bg=self.color_bg,)
        self.c2 = Canvas(self.master,width=512,height=512,bg=self.color_bg,)
        self.c.pack(fill=BOTH,expand=True,side = LEFT)
        self.c2.pack(fill=BOTH,expand=True,side = RIGHT)
        menu = Menu(self.master)
        self.master.config(menu=menu)
        filemenu = Menu(menu)
        colormenu = Menu(menu)
        menu.add_cascade(label='File',menu=colormenu)
        colormenu.add_command(label='Load',command=lambda:open_file_2(self.c))
        optionmenu = Menu(menu)
        menu.add_cascade(label='Options',menu=optionmenu)
        optionmenu.add_command(label='Clear Canvas',command=self.clear)
        optionmenu.add_command(label='Exit',command=self.master.destroy) 
        button4=Button(root,fg="green",text="convert",command=lambda:getter(self.c,self.c2))
        button4.pack(side=RIGHT)
        self.master.geometry('+10+10')
        
    def c_musin(self):  #changing the pen color
        self.color_fg="#2365ad"
    def c_epi_mature(self):  #changing the pen color
        self.color_fg="#30b155"
    def c_epi(self):  #changing the pen color
        self.color_fg="#4c86b2"
    def c_reactive_stroma(self):  #changing the pen color
        self.color_fg="#842e1e"
    def c_necrosis(self):  #changing the pen color
        self.color_fg="#c26744"
    def c_blood(self):  #changing the pen color
        self.color_fg="#ea242e"
    def c_inf(self):  #changing the pen color
        self.color_fg="#4f3286"
    def stroma(self):  #changing the pen color
        self.color_fg="#907e99"
    def c_muscle(self):  #changing the pen color
        self.color_fg="#d7c2d8"
    def c_loos_st(self):  #changing the pen color
        self.color_fg="#9d3b8a"        
    def drawWidgets3(self,mode):
        self.controls = Frame(self.master,padx = 5,pady = 5)
        Label(self.controls, text='Pen Width:',font=('arial 12')).grid(row=0,column=0)
        self.slider = ttk.Scale(self.controls,from_= 1, to = 200,command=self.changeW,orient=HORIZONTAL,length = 200)
        self.slider.set(self.penwidth)
        self.slider.grid(row=0,column=1)
        self.controls.pack(side=TOP)
        
        self.controls2 = Frame(self.master)

        self.c1=Button(self.controls2,fg = 'black',bg="white",relief = "flat",font = 'arial 18',text="background",command = self.background,width=10)
        self.c1.grid(row=1,column = 0)
        self.c1=Button(self.controls2,fg = 'black',bg="#30b155",relief = "flat",font = 'arial 18',text="epi mature",command = self.c_epi_mature,width=10)
        self.c1.grid(row=2,column = 0)
        self.c1=Button(self.controls2,fg = 'black',bg="#2365ad",relief = "flat",font = 'arial 18',text="musin",command = self.c_musin,width=10)
        self.c1.grid(row=0,column = 0)
        
        self.c1=Button(self.controls2,fg = 'black',bg="#4c86b2",relief = "flat",font = 'arial 18',text="epi",command = self.c_epi,width=10)
        self.c1.grid(row=3,column = 0)
        self.c1=Button(self.controls2,fg = 'black',bg="#c26744",relief = "flat",font = 'arial 18',text="necrosis",command = self.c_necrosis,width=10)
        self.c1.grid(row=4,column = 0)
        self.c1=Button(self.controls2,fg = 'black',bg="#842e1e",relief = "flat",font = 'arial 18',text="reactive stroma",command = self.c_reactive_stroma,width=10)
        self.c1.grid(row=5,column = 0)
        
        self.c1=Button(self.controls2,fg = 'black',bg="#ea242e",relief = "flat",font = 'arial 18',text="blood",command = self.c_blood,width=10)
        self.c1.grid(row=6,column = 0)
        self.c1=Button(self.controls2,fg = 'black',bg="#4f3286",relief = "flat",font = 'arial 18',text="inf cells",command = self.c_inf,width=10)
        self.c1.grid(row=7,column = 0)
        self.c1=Button(self.controls2,fg = 'black',bg="#907e99",relief = "flat",font = 'arial 18',text="stroma",command = self.stroma,width=10)
        self.c1.grid(row=8,column = 0)
        
        self.c1=Button(self.controls2,fg = 'black',bg="#d7c2d8",relief = "flat",font = 'arial 18',text="muscle",command = self.c_muscle,width=10)
        self.c1.grid(row=9,column = 0)
        self.c1=Button(self.controls2,fg = 'black',bg="#9d3b8a",relief = "flat",font = 'arial 18',text="loose stroam",command = self.c_loos_st,width=10)
        self.c1.grid(row=10,column = 0)

        
        self.controls2.pack(side=LEFT, anchor=NW)
        
        
        self.c = Canvas(self.master,width=512,height=512,bg=self.color_bg,)
        self.c2 = Canvas(self.master,width=512,height=512,bg=self.color_bg,)
        self.c.pack(fill=BOTH,expand=True,side = LEFT)
        self.c2.pack(fill=BOTH,expand=True,side = RIGHT)
        menu = Menu(self.master)
        self.master.config(menu=menu)
        filemenu = Menu(menu)
        colormenu = Menu(menu)
        menu.add_cascade(label='File',menu=colormenu)
        colormenu.add_command(label='Load',command=lambda:open_file_3(self.c))
        optionmenu = Menu(menu)
        menu.add_cascade(label='Options',menu=optionmenu)
        optionmenu.add_command(label='Clear Canvas',command=self.clear)
        optionmenu.add_command(label='Exit',command=self.master.destroy) 
        button4=Button(root,fg="green",text="convert",command=lambda:getter(self.c,self.c2))
        button4.pack(side=RIGHT)
        self.master.geometry('+10+10')
        
        

        
        

if __name__ == '__main__':
    
    root = Tk()
    main(root)
    root.title('Synthesizing histological images GUI') 
    root.withdraw()
    root.mainloop()
