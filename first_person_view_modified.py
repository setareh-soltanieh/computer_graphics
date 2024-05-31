# This code is for changing the first person's camera view with direct inputs

import glfw
import glfw.GLFW as GLFW_CONSTANTS
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram,compileShader
import numpy as np
import pyrr

pose = None
ori = None

def createShader(vertexFilepath: str, fragmentFilepath: str) -> int:    

    with open(vertexFilepath,'r') as f:
        vertex_src = f.readlines()

    with open(fragmentFilepath,'r') as f:
        fragment_src = f.readlines()
    
    shader = compileProgram(compileShader(vertex_src, GL_VERTEX_SHADER),
                            compileShader(fragment_src, GL_FRAGMENT_SHADER))
    
    return shader

class Entity:

    def __init__(self, position, eulers):

        self.position = np.array(position, dtype=np.float32)
        self.eulers = np.array(eulers, dtype=np.float32)
    
    def get_model_transform(self) -> np.ndarray:

        model_transform = pyrr.matrix44.create_identity(dtype=np.float32)
        
        model_transform = pyrr.matrix44.multiply(
            m1=model_transform, 
            m2=pyrr.matrix44.create_from_y_rotation(
                theta = np.radians(self.eulers[2]), 
                dtype=np.float32
            )
        )

        model_transform = pyrr.matrix44.multiply(
            m1=model_transform, 
            m2=pyrr.matrix44.create_from_translation(
                vec=self.position,
                dtype=np.float32
            )
        )

        return model_transform

class Triangle(Entity):
    
    def update(self, rate: float) -> None:

        self.eulers[2] += rate * 0.25
        if self.eulers[2] > 360:
            self.eulers[2] -= 360

class Camera(Entity):

    def __init__(self, position, eulers):

        super().__init__(position, eulers)

        #the camera's three fundamental directions: forwards, up & right
        self.localUp = np.array([0,0,1], dtype=np.float32)
        self.localRight = np.array([0,1,0], dtype=np.float32)
        self.localForwards = np.array([1,0,0], dtype=np.float32)

        #directions after rotation
        self.up = np.array([0,0,1], dtype=np.float32)
        self.right = np.array([0,1,0], dtype=np.float32)
        self.forwards = np.array([1,0,0], dtype=np.float32)
    
    def calculate_vectors_cross_product(self) -> None:

        #calculate the forwards vector directly using spherical coordinates
        self.forwards = np.array(
            [
                np.cos(np.radians(self.eulers[2])) * np.cos(np.radians(self.eulers[1])),
                np.sin(np.radians(self.eulers[2])) * np.cos(np.radians(self.eulers[1])),
                np.sin(np.radians(self.eulers[1]))
            ],
            dtype=np.float32
        )
        self.right = pyrr.vector.normalise(np.cross(self.forwards, self.localUp))
        self.up = pyrr.vector.normalise(np.cross(self.right, self.forwards))
    
    def update(self) -> None:

        self.calculate_vectors_cross_product()
        
    def get_view_transform(self) -> np.ndarray:
        """ Return's the camera's view transform. """

        return pyrr.matrix44.create_look_at(
            eye = self.position,
            target = self.position + self.forwards,
            up = self.up,
            dtype = np.float32
        )

class Scene:

    def __init__(self):

        self.triangle = Triangle(
            position = [3,0,0],
            eulers = [0,0,90]
        )

        self.camera = Camera(
            position = [0,0,0],
            eulers = [0,0,0]
        )
    
    def update(self, rate: float) -> None:

        self.triangle.update(rate)
        self.camera.update()
    
    def move_camera(self, newPos: np.ndarray) -> None:

        self.camera.position = newPos
    
    def spin_camera(self, newEulers: np.ndarray) -> None:

        self.camera.eulers = newEulers

        #modular check: camera can spin full revolutions 
        # around z axis.
        if self.camera.eulers[2] < 0:
            self.camera.eulers[2] += 360
        elif self.camera.eulers[2] > 360:
            self.camera.eulers[2] -= 360
        
        #clamping: around the y axis (up and down),
        # we never want the camera to be looking fully up or down.
        self.camera.eulers[1] = min(89, max(-89, self.camera.eulers[1]))

class App:


    def __init__(self):

        self.set_up_glfw()        
        self.make_assets()
        self.set_onetime_uniforms()
        self.get_uniform_locations()
        self.set_up_input_systems()
        self.set_up_timer()
        self.mainLoop()
    
    def set_up_glfw(self) -> None:
        """ Set up the glfw environment """

        self.screenWidth = 640
        self.screenHeight = 480

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

    def make_assets(self) -> None:

        self.scene = Scene()
        self.triangle_mesh = TriangleMesh()
        self.shader = createShader("shaders3/vertex.txt", "shaders3/fragment.txt")
    
    def set_onetime_uniforms(self) -> None:

        glUseProgram(self.shader)

        projection_transform = pyrr.matrix44.create_perspective_projection(
            fovy = 45, 
            aspect = self.screenWidth / self.screenHeight, 
            near = 0.1, far = 10, dtype = np.float32
        )
        glUniformMatrix4fv(
            glGetUniformLocation(self.shader, "projection"), 
            1, GL_FALSE, projection_transform
        )
    
    def get_uniform_locations(self) -> None:

        glUseProgram(self.shader)
        self.modelMatrixLocation = glGetUniformLocation(self.shader,"model")
        self.viewMatrixLocation = glGetUniformLocation(self.shader, "view")
    
    def set_up_input_systems(self) -> None:

        glfw.set_input_mode(
            self.window, 
            GLFW_CONSTANTS.GLFW_CURSOR, 
            GLFW_CONSTANTS.GLFW_CURSOR_HIDDEN
        )
        glfw.set_cursor_pos(
            self.window,
            self.screenWidth // 2, 
            self.screenHeight // 2
        )

        #based on the combination of wasd,
        #an offset is applied to the camera's direction
        # when walking. w = 1, a = 2, s = 4, d = 8
        self.walk_offset_lookup = {
            1: 0,
            2: 90,
            3: 45,
            4: 180,
            6: 135,
            7: 90,
            8: 270,
            9: 315,
            11: 0,
            12: 225,
            13: 270,
            14: 180
        }
    
    def set_up_timer(self) -> None:
        self.lastTime = glfw.get_time()
        self.currentTime = 0
        self.numFrames = 0
        self.frameTime = 0
    
    def mainLoop(self) -> None:

        glClearColor(0.1, 0.2, 0.2, 1)
        (w,h) = glfw.get_framebuffer_size(self.window)
        glViewport(0,0,w, h)
        running = True

        while (running):

            #check events
            if glfw.window_should_close(self.window) \
                or glfw.get_key(self.window, GLFW_CONSTANTS.GLFW_KEY_ESCAPE) == GLFW_CONSTANTS.GLFW_PRESS:
                running = False

            glfw.poll_events()
            
            self.update_camera_pose()

            #update scene
            self.scene.update(self.frameTime / 16.667)
            
            #refresh screen
            glClear(GL_COLOR_BUFFER_BIT)
            glUseProgram(self.shader)

            glUniformMatrix4fv(
                self.viewMatrixLocation, 
                1, GL_FALSE, 
                self.scene.camera.get_view_transform()
            )

            glUniformMatrix4fv(
                self.modelMatrixLocation,
                1,GL_FALSE,
                self.scene.triangle.get_model_transform()
            )
            glBindVertexArray(self.triangle_mesh.vao)
            glDrawArrays(GL_TRIANGLES, 0, self.triangle_mesh.vertex_count)

            glFlush()

            #timing
            self.calcuateFramerate()

        self.quit()
    
    def update_camera_pose(self):
        global pose
        global ori 

        if pose is None:
            pose = np.array([0, 0, 0], np.float32)
        else:
            pose += np.array([0, 0, 0.001], np.float32)
        
        self.scene.move_camera(pose)

        if ori is None:
            ori = np.array([0, 0, 0], np.float32)
        else:
            ori += np.array([0, 0, 0], np.float32)

        self.scene.spin_camera(ori)
    
    def calcuateFramerate(self) -> None:
        """
            Calculate the framerate and frametime
        """

        self.currentTime = glfw.get_time()
        delta = self.currentTime - self.lastTime
        if (delta >= 1):
            framerate = int(self.numFrames/delta)
            glfw.set_window_title(self.window, f"Running at {framerate} fps.")
            self.lastTime = self.currentTime
            self.numFrames = -1
            self.frameTime = float(1000.0 / max(60,framerate))
        self.numFrames += 1
    
    def quit(self) -> None:
        """ Free any allocated memory """

        self.triangle_mesh.destroy()
        glDeleteProgram(self.shader)
        glfw.terminate()

class TriangleMesh:


    def __init__(self):

        # x, y, z, r, g, b
        self.vertices = (
            -0.5, -0.5, 0.0, 1.0, 0.0, 0.0,
             0.5, -0.5, 0.0, 0.0, 1.0, 0.0,
             0.0,  0.5, 0.0, 0.0, 0.0, 1.0
        )
        self.vertices = np.array(self.vertices, dtype=np.float32)

        self.vertex_count = 3

        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)
        self.vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, self.vertices.nbytes, self.vertices, GL_STATIC_DRAW)

        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(0))
        
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(12))

    def destroy(self):
        
        glDeleteVertexArrays(1, (self.vao,))
        glDeleteBuffers(1,(self.vbo,))

myApp = App()