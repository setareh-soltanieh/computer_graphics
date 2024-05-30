import glfw
import glfw.GLFW as GLFW_CONSTANTS
from OpenGL.GL import *

SCREEN_WIDTH = 640
SCREEN_HEIGHT = 480

class App:

    def __init__(self):
        """ Initialise the program """
        self.set_up_glfw()
        glClearColor(0.1, 0.2, 0.2, 1)

    def set_up_glfw(self) -> None:
        """ Set up the glfw environment """

        self.screenWidth = SCREEN_WIDTH
        self.screenHeight = SCREEN_HEIGHT

        glfw.init()
        glfw.window_hint(GLFW_CONSTANTS.GLFW_CONTEXT_VERSION_MAJOR,3)
        glfw.window_hint(GLFW_CONSTANTS.GLFW_CONTEXT_VERSION_MINOR,3)
        glfw.window_hint(
            GLFW_CONSTANTS.GLFW_OPENGL_PROFILE, 
            GLFW_CONSTANTS.GLFW_OPENGL_CORE_PROFILE
        )
        glfw.window_hint(
            GLFW_CONSTANTS.GLFW_OPENGL_FORWARD_COMPAT, 
            GLFW_CONSTANTS.GLFW_TRUE
        )
        glfw.window_hint(GLFW_CONSTANTS.GLFW_DOUBLEBUFFER, False)
        self.window = glfw.create_window(
            self.screenWidth, self.screenHeight, "Title", None, None
        )
        glfw.make_context_current(self.window)

    def run(self):
        """ Run the app """

        while not glfw.window_should_close(self.window):
            #check events
            if glfw.get_key(self.window, GLFW_CONSTANTS.GLFW_KEY_ESCAPE) \
                == GLFW_CONSTANTS.GLFW_PRESS:
                break
            glfw.poll_events()
            #refresh screen
            glClear(GL_COLOR_BUFFER_BIT)
            glfw.swap_buffers(self.window)

    def quit(self):
        """ cleanup the app, run exit code """
        glfw.destroy_window(self.window)
        glfw.terminate()

my_app = App()
my_app.run()
my_app.quit()


