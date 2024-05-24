import pygame as pg 
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import numpy as np
import pyrr 

def createShader(vertexFilepath, fragmentFilepath):

    with open(vertexFilepath, 'r') as f:
        vertex_src = f.readlines()

    with open(fragmentFilepath, 'r') as f:
        fragment_src = f.readlines()
    
    shader = compileProgram(compileShader(vertex_src, GL_VERTEX_SHADER), compileShader(fragment_src, GL_FRAGMENT_SHADER))

    return shader


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

        self.triangle = Entity(
            position = [0.0,0,0],
            eulers = [0,0,0]
        )

        self.triangle_mesh = TriangleMesh()
        self.shader = createShader("shaders/vertex.txt", "shaders/fragment.txt")

        self.main_loop()

    def main_loop(self):

        running = True
        while(running):
            for event in pg.event.get():
                if (event.type == pg.QUIT):
                    running = False
            
            self.triangle.eulers[2] += 0.25
            if self.triangle.eulers[2] > 360:
                self.triangle.eulers[2] -= 360

            glClear(GL_COLOR_BUFFER_BIT)
    
            glUseProgram(self.shader)

            model_transform = pyrr.matrix44.create_identity(dtype=np.float32)

            model_transform = pyrr.matrix44.multiply(
                m1 = model_transform,
                m2 = pyrr.matrix44.create_from_y_rotation(
                    theta=np.radians(self.triangle.eulers[2]), 
                    dtype=np.float32
                )
            )

            self.triangle_mesh.build_vertices(model_transform)
            glBindVertexArray(self.triangle_mesh.vao)
            glDrawArrays(GL_TRIANGLES, 0, self.triangle_mesh.vertex_count)

            pg.display.flip()

            self.clock.tick(60)
        self.quit()
    
    def quit(self):

        pg.quit()

class Entity:
    """ Basic description of anything which has a position and rotation"""


    def __init__(self, position, eulers):

        self.position = np.array(position, dtype=np.float32)
        self.eulers = np.array(eulers, dtype=np.float32)

class Triangle():

    def __init__(self):
        # negative x is left side of screen | positive x is right side of screen
        # negative y is bottom of screen | positive y is top of screen
        # negative z is in distance | positive Z is closer 

        # x, y, z
        self.vertices = (
            -0.5, -0.5, 0.0, 1.0, 0.0, 0.0,
             0.5, -0.5, 0.0, 0.0, 1.0, 0.0,
             0.0,  0.5, 0.0, 0.0, 0.0, 1.0,
        )

        self.vertices = np.array(self.vertices, dtype=np.float32)
        self.vertices_count = 3

        # generate 1 vertex array object 
        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)

        # generate 1 vertex buffer object (vbo stores data)
        self.vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        upload_ready = True
        if upload_ready:
            # we are sending data to buffer
            # The number of data is specified with self.vertices.nbytes and these data are field with self.vertices
            # since we are sending our data as array_buffers and we have bound the array buffer with the vbo,
            # we are sending our data into vbo
            glBufferData(GL_ARRAY_BUFFER, self.vertices.nbytes, self.vertices, GL_STATIC_DRAW)
        
        print(f"Vertex Array handle: {self.vao}")
        print(f"Buffer handle: {self.vbo}")
        buffer_size = glGetBufferParameteriv(GL_ARRAY_BUFFER, GL_BUFFER_SIZE)
        print(f"Our buffer is taking up {buffer_size} bytes is memory")
        
        # this is for the vertex attributes in the memory which has 3 elements x, y, z
        attribute_index = 0
        elements_per_attribute = 3
        element_type = GL_FLOAT
        normalized = GL_FALSE
        stride_in_bytes = 24 
        offset_in_bytes = 0
        
        glEnableVertexAttribArray(attribute_index)
        glVertexAttribPointer(
            attribute_index, elements_per_attribute, element_type, normalized, stride_in_bytes, ctypes.c_void_p(offset_in_bytes)
        )

        # this is for the color attributes in the memory which has 3 elements r, g, b
        attribute_index = 1
        elements_per_attribute = 3
        element_type = GL_FLOAT
        normalized = GL_FALSE
        stride_in_bytes = 24
        offset_in_bytes = 12
        
        glEnableVertexAttribArray(attribute_index)
        glVertexAttribPointer(
            attribute_index, elements_per_attribute, element_type, normalized, stride_in_bytes, ctypes.c_void_p(offset_in_bytes)
        )


class TriangleMesh:

    def __init__(self):

        # self.originalPositions = (
        #     pyrr.vector4.create(-0.5, -0.5, 0.0, 1.0, dtype=np.float32),
        #     pyrr.vector4.create( 0.5, -0.5, 0.0, 1.0, dtype=np.float32),
        #     pyrr.vector4.create( 0.0,  0.5, 0.0, 1.0, dtype=np.float32)
        # )
        # self.originalColors = (
        #     pyrr.vector3.create(1.0, 0.0, 0.0, dtype=np.float32),
        #     pyrr.vector3.create(0.0, 1.0, 0.0, dtype=np.float32),
        #     pyrr.vector3.create(0.0, 0.0, 1.0, dtype=np.float32)
        # )
        
        self.originalPosition = (
            pyrr.vector4.create(-0.5, -0.5, 0.0, 1.0, dtype=np.float32),
            pyrr.vector4.create( 0.5, -0.5, 0.0, 1.0, dtype=np.float32),
            pyrr.vector4.create( 0.0,  0.5, 0.0, 1.0, dtype=np.float32)
        )
        self.originalColor = (
            pyrr.vector3.create(1.0, 0.0, 0.0, dtype=np.float32),
            pyrr.vector3.create(0.0, 1.0, 0.0, dtype=np.float32),
            pyrr.vector3.create(0.0, 0.0, 1.0, dtype=np.float32)
        )

        self.vertex_count = 3
        self.vao = glGenVertexArrays(1)
        self.vbo = glGenBuffers(1)

        self.build_vertices(pyrr.matrix44.create_identity(dtype=np.float32))

        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(0))

        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(12))


    def build_vertices(self, transform):
        
        self.vertices = np.array([], dtype=np.float32)

        for i in range(self.vertex_count):

            transformed_position = pyrr.matrix44.multiply(
                m1 = self.originalPosition[i],
                m2 = transform
            )

            self.vertices = np.append(self.vertices, transformed_position[0:3])
            self.vertices = np.append(self.vertices, self.originalColor[i])
        
        glBindVertexArray(self.vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, self.vertices.nbytes, self.vertices, GL_STATIC_DRAW)
    
    def destroy(self):

        glDeleteVertexArrays(1, (self.vao,))
        glDeleteBuffers(1, (self.vbo,))



if __name__ == "__main__":
    myApp = App()

