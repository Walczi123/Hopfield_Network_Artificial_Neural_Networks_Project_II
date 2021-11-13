from tkinter import *
from sys import platform
if platform == "darwin":
    from tkmacosx import Button


class Board:

    def __init__(self, cell_size: int, dim_size: int):
        self.size = cell_size * dim_size
        self.dim_size = dim_size
        self.prepare_window()
        self.prepare_controls()
        self.prepare_board()
        self.prepare_config_panel()

    def prepare_window(self):
        self.window = Tk()
        self.window.title('Hopfield')
        self.window.resizable(width=self.size, height=self.size)
        self.window.minsize(self.size, int(3/4*self.size))
        Grid.rowconfigure(self.window, 0, weight=1)
        Grid.columnconfigure(self.window, 0, weight=3)

    def prepare_controls(self):
        self.buttons = []

    def prepare_board(self):
        board_frame = Frame(self.window, width=self.size, height=self.size)
        board_frame.grid(row=0, column=0, sticky=N+S+E+W)
        for row_index in range(self.dim_size):
            Grid.rowconfigure(board_frame, row_index, weight=1)
            buttons_row = []
            for col_index in range(self.dim_size):
                Grid.columnconfigure(board_frame, col_index, weight=1)
                btn = Button(board_frame, highlightbackground='white', background="white",activebackground="white",activeforeground="white",
                             command=lambda r=row_index, c=col_index: self.button_handler(r, c))
                btn.grid(row=row_index, column=col_index, sticky=N+S+E+W)
                buttons_row.append(btn)
            self.buttons.append(buttons_row)

    def prepare_config_panel(self):
        Grid.rowconfigure(self.window, 0, weight=1)
        Grid.columnconfigure(self.window, 1, weight=1)
        frame = Frame(self.window, width=100)
        frame.grid(row=0, column=2, sticky=N+S+E+W)

    def button_handler(self, row: int, col: int):
        self.window.update()
        if self.buttons[row][col]["highlightbackground"] == "black":
            self.set_color(row, col, "white")
        else:
            self.set_color(row, col, "black")
    
    def overwrite_board(self, grid):
        for row in range(self.dim_size):
            for col in range(self.dim_size):
                if grid[row, col] == 1:
                    self.set_color(row, col, "black")
                else:
                    self.set_color(row, col, "white")
    
    def set_color(self, row, col, color):
        self.buttons[row][col]["highlightbackground"] = color
        self.buttons[row][col]["bg"] = color
        self.buttons[row][col]["activebackground"] = color
        self.buttons[row][col]["activeforeground"] = color

    def run(self):
        self.window.mainloop()


if __name__ == "__main__":
    board = Board(100,5)
    board.run()
