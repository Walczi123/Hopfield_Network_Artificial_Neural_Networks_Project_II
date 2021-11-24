import copy
import tkinter as tk

from math import floor

import numpy as np

class UI(tk.Frame):
    def __init__(self, size=32, color1="white", color2="black"):
        self.size = size
        self.color1 = color1
        self.color2 = color2
        self.pieces = {}

        self.canvas_width = 600
        self.canvas_height = 600

    def init_window(self, parent):
        tk.Frame.__init__(self, parent)
        self.canvas = tk.Canvas(self, borderwidth=0, highlightthickness=0,
                                width=self.canvas_width, height=self.canvas_height, background="gray")
        self.canvas.pack(side="top", fill="both", expand=True, padx=2, pady=2)

        self.canvas.bind("<Configure>", self.refresh)
        self.canvas.bind("<Button-1>", self.clickHandle)
        self.canvas.focus_set()

    def refresh(self, event):
        xsize = int((event.width-1) / self.columns)
        ysize = int((event.height-1) / self.rows)
        self.size = min(xsize, ysize)
        
        self.update()

    def update(self):  
        self.canvas.delete("square")
        for row in range(self.rows):
            for col in range(self.columns):
                x1 = (col * self.size)
                y1 = (row * self.size)
                x2 = x1 + self.size 
                y2 = y1 + self.size
                self.canvas.create_rectangle(x1, y1, x2, y2, outline="black", fill=self.color_field(row,col), tags="square")

        # self.canvas.tag_raise("piece")
        # self.canvas.tag_lower("square")

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
        array = [[1, -1, -1],[ 1, 1, -1],[ -1, 1, 1]]
        array = [1] * 10000
        size = (100,100)
        array = np.array(array).reshape(size)

        root = tk.Tk()

        self.board = copy.deepcopy(array)
        self.rows = size[0]
        self.columns = size[1]

        self.init_window(root)
        self.pack(side="top", fill="both", expand="true", padx=4, pady=4)
        root.mainloop()

if __name__ == "__main__":
    ui = UI()
    ui.run()
    # a = [1] * 100
    # print(a)