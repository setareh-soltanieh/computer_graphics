import pygame as pg 
from OpenGL.GL import *

class App: 
    
    def __init__(self):
        
        pg.init()

        # openGL version: 3.3 - Profile: Core
        major_version = 3
        minor_version = 3
        img_width = 640
        img_height = 480
        profile = pg.GL_CONTEXT_PROFILE_CORE

        pg.display.gl_set_attribute(pg.GL_CONTEXT_MAJOR_VERSION, major_version)
        pg.display.gl_set_attribute(pg.GL_CONTEXT_MINOR_VERSION, minor_version)
        pg.display.gl_set_attribute(pg.GL_CONTEXT_PROFILE_MASK, profile)

        self.scree_size = (img_width, img_height)
        pg.display.set_mode(self.scree_size, pg.OPENGL|pg.DOUBLEBUF)
        self.clock = pg.time.Clock()

        glClearColor(0.45, 0.83, 0.24, 1.0)

        self.main_loop()

    def main_loop(self):

        running = True
        while(running):
            for event in pg.event.get():
                if (event.type == pg.QUIT):
                    running = False
            
            glClear(GL_COLOR_BUFFER_BIT)

            pg.display.flip()

            self.clock.tick(60)
        self.quit()
    
    def quit(self):

        pg.quit()

if __name__ == "__main__":
    myApp = App()

