import copy
import os
import tkinter as tk
from tkinter import Frame, ttk, filedialog
from tkinter.constants import NO
from tkinter.filedialog import askopenfile

from math import floor
from common.common_functions import array_to_vector, data_to_array, get_size_from_filename, read_data_as_vectors

import numpy as np

from hopfield import HopfieldNetwork

class UI(tk.Frame):
    def __init__(self, size=32, color1="white", color2="black"):
        self.size = size
        self.color1 = color1
        self.color2 = color2
        self.pieces = {}
        # self.rows = 1
        # self.columns = 1

        self.canvas_width = 400
        self.canvas_height = 400

        self.board = []
        self.set_board([1],(1,1))
        self.vectors = []

        self.label_loaded = None

        self.parent = tk.Tk()

    def set_board(self, vector, size, update = False): 
        self.rows = size[0]
        self.columns = size[1]
        self.board = data_to_array(vector, size) 
        if update:
            self.refresh()


    def init_window(self):
        tk.Frame.__init__(self, self.parent)
        self.canvas = tk.Canvas(self, borderwidth=0, highlightthickness=0,
                                width=self.canvas_width, height=self.canvas_height, background="gray")
        self.canvas.pack(side="top", fill="both", expand=True, padx=2, pady=2)

        self.canvas.bind("<Configure>", self.refresh)
        self.canvas.bind("<Button-1>", self.clickHandle)
        self.canvas.focus_set()

        # screen_width = self.parent.winfo_width()
        # screen_height = self.parent.winfo_height()

        # self.parent.minsize(screen_width, screen_height)

    def refresh(self, event = None):
        if event is None:
            xsize = int((self.canvas.winfo_width()-1) / self.columns)
            ysize = int((self.canvas.winfo_height()-1) / self.rows)
        else:
            xsize = int((event.width-1) / self.columns)
            ysize = int((event.height-1) / self.rows)
        self.size = min(xsize, ysize)
        
        self.update()

    def update(self):  
        if len(self.canvas.find_withtag("square")) != 0:
            self.canvas.delete("square")
        for row in range(self.rows):
            for col in range(self.columns):
                x1 = (col * self.size)
                y1 = (row * self.size)
                x2 = x1 + self.size 
                y2 = y1 + self.size
                self.canvas.create_rectangle(x1, y1, x2, y2, outline="black", fill=self.color_field(row,col), tags="square")

    def draw_top_frame(self):
        top = Frame(self.parent)
        top.pack(side='top')
        top2 = Frame(self.parent)
        top2.pack(side='top')
        self.top2 = top2  

        # Create a Button
        browse_button = tk.Button(self.parent, text="Browse", command=self.open_file)

        # label.pack(in_=top, side='left')
        browse_button.pack(in_=top, side='left')      
        
    def draw_bottom_frame(self):
        bot = Frame(self.parent)
        bot.pack(side='bottom')

        # create a combobox
        self.combobox = tk.StringVar()
        self.vector_cb = ttk.Combobox(self.parent, textvariable=self.combobox)
        self.vector_cb['values'] = [m for m in range(len(self.vectors))]
        self.vector_cb['state'] = 'readonly'

        # place the widget
        # month_cb.pack(fill=tk.X, padx=5, pady=5)
        self.vector_cb.pack(in_=bot, side='left') 

        self.vector_cb.bind('<<ComboboxSelected>>', self.vector_changed)

        bot2 = Frame(self.parent)
        bot2.pack(side='bottom')

         # Create a Button
        self.train_button = tk.Button(self.parent, text="Train", command=self.train)

        # label.pack(in_=top, side='left')
        self.train_button.pack(in_=bot2, side='left') 
        self.train_button["state"] = "disabled"

         # Create a Button
        self.predict_button = tk.Button(self.parent, text="Predict", command=self.predict)

        # label.pack(in_=top, side='left')
        self.predict_button.pack(in_=bot2, side='right')  
        self.predict_button["state"] = "disabled"

    def train(self):
        vec = array_to_vector(self.board)
        self.nn.train(vec)
        print('trained')

    def predict(self):
        vec = array_to_vector(self.board)
        vec = self.nn.predict(vec, 1)
        self.board = data_to_array(vec, self.board_size)
        self.update()
        print('predicted')

    def open_file(self):
        file = filedialog.askopenfile(mode='r', filetypes=[('Prepared vectors sets', '*.csv')])
        if file:
            filepath = os.path.abspath(file.name)
            self.filename = os.path.basename(filepath)
            if self.label_loaded is None:
                self.label_loaded = tk.Label(self.parent, text="The loaded file: " + str(self.filename), font=('Aerial 11'))
                self.label_loaded.pack(in_=self.top2, side='bottom')   
            else:
                self.label_loaded['text']="The loaded file: " + str(self.filename)
            self.vectors = read_data_as_vectors(filepath)
            
            self.vector_cb['values'] = [m for m in range(len(self.vectors))]
            self.vector_changed(0)

            self.train_button["state"] = "normal"
            self.predict_button["state"] = "normal"

            self.nn = HopfieldNetwork(self.board_size[0]*self.board_size[1])

    def vector_changed(self, event = None):
        if type(event) is not int:
            event = int(self.vector_cb.get())
        self.vector_cb.current(event)
        self.board_size = get_size_from_filename(self.filename)
        self.set_board(self.vectors[event],self.board_size,True)
        self.selected_vector = event

    def color_field(self, row, col):
        field_value = self.board[row][col]
        if field_value == -1:
            return self.color1
        else: 
            return self.color2

    def color_rectangle(self, x, y):
        
        col = floor((x ) / self.size)
        row = floor((y ) / self.size)
        if self.columns > col >= 0 and self.rows > row >= 0 :
            self.board[row][col] = -self.board[row][col]

            rectangle_x = (col * self.size)
            rectangle_y = (row * self.size)
            rectangle_x2 = rectangle_x + self.size
            rectangle_y2 = rectangle_y + self.size

            self.canvas.create_rectangle(rectangle_x, rectangle_y, rectangle_x2, rectangle_y2, outline="black", fill=self.color_field(row,col), tags="square")


    def clickHandle(self, event):

        xMouse = event.x
        yMouse = event.y
        
        self.color_rectangle(xMouse, yMouse)

    def run(self):
        # array = [[1, -1, -1],[ 1, 1, -1],[ -1, 1, 1]]
        # array = [1] * 10000
        # size = (3,3)
        # array = np.array(array).reshape(size)
        # self.set_board(array, size)

        

        # self.board = copy.deepcopy(array)
        # self.rows = size[0]
        # self.columns = size[1]

        self.init_window()
        self.draw_top_frame()
        self.pack(side="top", fill="both", expand="true", padx=4, pady=4)
        self.draw_bottom_frame()
        # button1 = tk.Button(root, text='Yes', command=lambda:function('Yes'))
        # button1.pack()
        self.parent.mainloop()

if __name__ == "__main__":
    ui = UI()
    ui.run()
    # a = [1] * 100
    # print(a)