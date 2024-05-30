import glfw
import glfw.GLFW as GLFW_CONSTANTS
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import numpy as np
import pyrr

pose = None
ori = None

vertex_shader_source = """
#version 330 core
layout (location = 0) in vec3 aPos;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main()
{
    gl_Position = projection * view * model * vec4(aPos, 1.0);
}
"""

fragment_shader_source = """
#version 330 core
out vec4 FragColor;

void main()
{
    FragColor = vec4(1.0, 1.0, 1.0, 1.0);
}
"""

class Entity:
    def __init__(self, position, eulers):
        self.position = np.array(position, dtype=np.float32)
        self.eulers = np.array(eulers, dtype=np.float32)

    def get_model_transform(self) -> np.ndarray:
        model_transform = pyrr.matrix44.create_identity(dtype=np.float32)
        model_transform = pyrr.matrix44.multiply(
            m1=model_transform,
            m2=pyrr.matrix44.create_from_translation(
                vec=self.position,
                dtype=np.float32
            )
        )
        return model_transform

class Cube(Entity):
    def update(self, rate: float) -> None:
        self.eulers[2] += rate * 0.25
        if self.eulers[2] > 360:
            self.eulers[2] -= 360

class Camera(Entity):
    def __init__(self, position, eulers):
        super().__init__(position, eulers)
        self.up = np.array([0, 0, 1], dtype=np.float32)

    def get_view_transform(self) -> np.ndarray:
        return pyrr.matrix44.create_look_at(
            eye=self.position,
            target=self.position + self.forward,
            up=self.up,
            dtype=np.float32
        )

class Scene:
    def __init__(self):
        self.cube = Cube(
            position=[0, 0, 0],
            eulers=[0, 0, 0]
        )
        self.camera = Camera(
            position=[3, 3, 3],
            eulers=[-45, 45, 0]
        )

    def update(self, rate: float) -> None:
        self.cube.update(rate)
        self.camera.update()

class App:
    def __init__(self):
        self.set_up_glfw()
        self.shader_program = self.create_shader_program()
        self.create_cube()
        self.make_assets()
        self.set_onetime_uniforms()
        self.get_uniform_locations()
        self.set_up_input_systems()
        self.set_up_timer()
        self.mainLoop()

    def set_up_glfw(self) -> None:
        self.screenWidth = 800
        self.screenHeight = 600
        if not glfw.init():
            raise Exception("GLFW can not be initialized!")
        glfw.window_hint(GLFW_CONSTANTS.GLFW_CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(GLFW_CONSTANTS.GLFW_CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(GLFW_CONSTANTS.GLFW_OPENGL_PROFILE, GLFW_CONSTANTS.GLFW_OPENGL_CORE_PROFILE)
        glfw.window_hint(GLFW_CONSTANTS.GLFW_OPENGL_FORWARD_COMPAT, GLFW_CONSTANTS.GLFW_TRUE)
        self.window = glfw.create_window(self.screenWidth, self.screenHeight, "Cube Visualization", None, None)
        if not self.window:
            glfw.terminate()
            raise Exception("GLFW window can not be created!")
        glfw.make_context_current(self.window)

    def create_shader_program(self):
        vertex_shader = compileShader(vertex_shader_source, GL_VERTEX_SHADER)
        fragment_shader = compileShader(fragment_shader_source, GL_FRAGMENT_SHADER)
        shader_program = compileProgram(vertex_shader, fragment_shader)
        return shader_program

    def create_cube(self):
        self.vertices = np.array([
            # Front face
            -0.5, -0.5,  0.5,
             0.5, -0.5,  0.5,
             0.5,  0.5,  0.5,
            -0.5,  0.5,  0.5,
            # Back face
            -0.5, -0.5, -0.5,
             0.5, -0.5, -0.5,
             0.5,  0.5, -0.5,
            -0.5,  0.5, -0.5,
            # Top face
             0.5,  0.5,  0.5,
            -0.5,  0.5,  0.5,
            -0.5,  0.5, -0.5,
             0.5,  0.5, -0.5,
            # Bottom face
            -0.5, -0.5,  0.5,
             0.5, -0.5,  0.5,
             0.5, -0.5, -0.5,
            -0.5, -0.5, -0.5,
            # Right face
             0.5, -0.5,  0.5,
             0.5,  0.5,  0.5,
             0.5,  0.5, -0.5,
             0.5, -0.5, -0.5,
            # Left face
            -0.5, -0.5,  0.5,
            -0.5,  0.5,  0.5,
            -0.5,  0.5, -0,
            -0.5, -0.5, -0.5,
            -0.5, -0.5, -0.5
        ], dtype=np.float32)

        self.indices = np.array([
            0, 1, 2, 2, 3, 0,  # Front face
            4, 5, 6, 6, 7, 4,  # Back face
            8, 9, 10, 10, 11, 8,  # Top face
            12, 13, 14, 14, 15, 12,  # Bottom face
            16, 17, 18, 18, 19, 16,  # Right face
            20, 21, 22, 22, 23, 20  # Left face
        ], dtype=np.uint32)

        self.VAO = glGenVertexArrays(1)
        glBindVertexArray(self.VAO)

        self.VBO = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.VBO)
        glBufferData(GL_ARRAY_BUFFER, self.vertices.nbytes, self.vertices, GL_STATIC_DRAW)

        self.EBO = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.EBO)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, self.indices.nbytes, self.indices, GL_STATIC_DRAW)

        # Position attribute
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GLfloat), ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)

        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)

    def make_assets(self) -> None:
        self.scene = Scene()

    def set_onetime_uniforms(self) -> None:
        glUseProgram(self.shader_program)
        projection_transform = pyrr.matrix44.create_perspective_projection(
            fovy=45,
            aspect=self.screenWidth / self.screenHeight,
            near=0.1, far=10, dtype=np.float32
        )
        glUniformMatrix4fv(
            glGetUniformLocation(self.shader_program, "projection"),
            1, GL_FALSE, projection_transform
        )

    def get_uniform_locations(self) -> None:
        pass  # No additional uniforms in this example

    def set_up_input_systems(self) -> None:
        pass  # No input systems for this example

    def set_up_timer(self) -> None:
        pass  # No timer setup for this example

    def mainLoop(self) -> None:
        glClearColor(0.1, 0.2, 0.2, 1)
        (w, h) = glfw.get_framebuffer_size(self.window)
        glViewport(0, 0, w, h)
        running = True
        while running:
            if glfw.window_should_close(self.window) or glfw.get_key(self.window, GLFW_CONSTANTS.GLFW_KEY_ESCAPE) == GLFW_CONSTANTS.GLFW_PRESS:
                running = False
            glfw.poll_events()
            self.update_camera_pose()
            self.scene.update(1.0)
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            glUseProgram(self.shader_program)
            view_transform = self.scene.camera.get_view_transform()
            glUniformMatrix4fv(
                glGetUniformLocation(self.shader_program, "view"),
                1, GL_FALSE, view_transform
            )
            model_transform = self.scene.cube.get_model_transform()
            glUniformMatrix4fv(
                glGetUniformLocation(self.shader_program, "model"),
                1, GL_FALSE, model_transform
            )
            glBindVertexArray(self.VAO)
            glDrawElements(GL_TRIANGLES, len(self.indices), GL_UNSIGNED_INT, None)
            glBindVertexArray(0)
            glfw.swap_buffers(self.window)
        self.quit()

    def update_camera_pose(self):
        pass  # No camera movement in this example

    def quit(self) -> None:
        glDeleteVertexArrays(1, [self.VAO])
        glDeleteBuffers(1, [self.VBO, self.EBO])
        glfw.terminate()


if __name__ == "__main__":
    app = App()

