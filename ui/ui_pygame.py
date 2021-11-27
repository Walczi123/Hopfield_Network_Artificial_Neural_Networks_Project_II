import pygame
from math import floor

from common.common_functions import data_to_array

class UI:
    def __init__(self):
        self.clock = pygame.time.Clock()
        self.screen_size = (600, 600)
        self.screen = pygame.display.set_mode(self.screen_size)
        
        self.screen.fill((0,0,0))

        self.set_board([1], (1,1))

    def set_board(self, vector, size): 
        self.board_size = size
        self.sqr_size = (int(self.screen_size[0]/self.board_size[0]), int(self.screen_size[1]/self.board_size[1]))
        self.board = data_to_array(vector, self.board_size)  

    def draw_board(self):
        for i in range(self.board_size[0]):
            for z in range(self.board_size[1]):
                if self.board[i, z] == 1:
                    pygame.draw.rect(self.screen, (255,255,255),[self.sqr_size[1]*(z),self.sqr_size[0]*(i),self.sqr_size[1],self.sqr_size[0]])
                else:
                    pygame.draw.rect(self.screen, (0,0,0), [self.sqr_size[1]*(z),self.sqr_size[0]*(i),self.sqr_size[1],self.sqr_size[0]])

    def click_on_board(self, mouse_pos):
        y = floor(mouse_pos[0]/self.sqr_size[0])
        x = floor(mouse_pos[1]/self.sqr_size[1])
        self.board[x][y] = - self.board[x][y]

    def handle_click(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.MOUSEBUTTONUP:
                pos = pygame.mouse.get_pos()
                self.click_on_board(pos)

        return True
    


    def run(self):
        array = [1]*10000
        size =(100,100)

        # self.board = data_to_array(array, size)

        self.set_board(array, size)

        pygame.init()
        pygame.display.set_caption("Hopfield")

        running = True
        while running:
            self.draw_board()

            pygame.display.update()
            self.clock.tick(30)

            running = self.handle_click()

        pygame.quit()

if __name__ == "__main__":
    ui = UI()
    ui.run()