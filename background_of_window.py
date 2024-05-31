import glfw
import glfw.GLFW as GLFW_CONSTANTS
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import numpy as np
import pyrr
from PIL import Image

pose = None
ori = None

vertex_shader_source = """
#version 330 core
layout (location = 0) in vec2 aPos;
layout (location = 1) in vec2 aTexCoord;

out vec2 TexCoord;

void main()
{
    gl_Position = vec4(aPos, 0.0, 1.0);
    TexCoord = aTexCoord;
}
"""

fragment_shader_source = """
#version 330 core
out vec4 FragColor;

in vec2 TexCoord;

uniform sampler2D texture1;

void main()
{
    FragColor = texture(texture1, TexCoord);
}
"""

triangle_vertex_shader_source = """
#version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aColor;

out vec3 ourColor;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main()
{
    gl_Position = projection * view * model * vec4(aPos, 1.0);
    ourColor = aColor;
}
"""

triangle_fragment_shader_source = """
#version 330 core
out vec4 FragColor;
in vec3 ourColor;

void main()
{
    FragColor = vec4(ourColor, 1.0f);
}
"""

def createShader(vertex_src, fragment_src):
    vertex_shader = compileShader(vertex_src, GL_VERTEX_SHADER)
    fragment_shader = compileShader(fragment_src, GL_FRAGMENT_SHADER)
    shader_program = compileProgram(vertex_shader, fragment_shader)
    return shader_program

class Entity:
    def __init__(self, position, eulers):
        self.position = np.array(position, dtype=np.float32)
        self.eulers = np.array(eulers, dtype=np.float32)

    def get_model_transform(self) -> np.ndarray:
        model_transform = pyrr.matrix44.create_identity(dtype=np.float32)
        model_transform = pyrr.matrix44.multiply(
            m1=model_transform,
            m2=pyrr.matrix44.create_from_y_rotation(
                theta=np.radians(self.eulers[2]),
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
        self.localUp = np.array([0, 0, 1], dtype=np.float32)
        self.localRight = np.array([0, 1, 0], dtype=np.float32)
        self.localForwards = np.array([1, 0, 0], dtype=np.float32)
        self.up = np.array([0, 0, 1], dtype=np.float32)
        self.right = np.array([0, 1, 0], dtype=np.float32)
        self.forwards = np.array([1, 0, 0], dtype=np.float32)

    def calculate_vectors_cross_product(self) -> None:
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
        return pyrr.matrix44.create_look_at(
            eye=self.position,
            target=self.position + self.forwards,
            up=self.up,
            dtype=np.float32
        )

class Scene:
    def __init__(self):
        self.triangle = Triangle(
            position=[5, 0, 0],
            eulers=[0, 0, 90]
        )
        self.camera = Camera(
            position=[0, 0, 0],
            eulers=[0, 0, 0]
        )

    def update(self, rate: float) -> None:
        self.triangle.update(rate)
        self.camera.update()

    def move_camera(self, newPos: np.ndarray) -> None:
        self.camera.position = newPos

    def spin_camera(self, newEulers: np.ndarray) -> None:
        self.camera.eulers = newEulers
        if self.camera.eulers[2] < 0:
            self.camera.eulers[2] += 360
        elif self.camera.eulers[2] > 360:
            self.camera.eulers[2] -= 360
        self.camera.eulers[1] = min(89, max(-89, self.camera.eulers[1]))

class App:
    def __init__(self):
        self.set_up_glfw()
        self.texture_id = self.load_texture("gfx/wood.jpeg")
        self.shader_program = self.create_shader_program()
        self.create_quad()
        self.make_assets()
        self.set_onetime_uniforms()
        self.get_uniform_locations()
        self.set_up_input_systems()
        self.set_up_timer()
        self.mainLoop()

    def set_up_glfw(self) -> None:
        self.screenWidth = 640
        self.screenHeight = 480
        if not glfw.init():
            raise Exception("GLFW can not be initialized!")
        glfw.window_hint(GLFW_CONSTANTS.GLFW_CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(GLFW_CONSTANTS.GLFW_CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(GLFW_CONSTANTS.GLFW_OPENGL_PROFILE, GLFW_CONSTANTS.GLFW_OPENGL_CORE_PROFILE)
        glfw.window_hint(GLFW_CONSTANTS.GLFW_OPENGL_FORWARD_COMPAT, GLFW_CONSTANTS.GLFW_TRUE)
        glfw.window_hint(GLFW_CONSTANTS.GLFW_DOUBLEBUFFER, True)
        self.window = glfw.create_window(self.screenWidth, self.screenHeight, "Title", None, None)
        if not self.window:
            glfw.terminate()
            raise Exception("GLFW window can not be created!")
        glfw.make_context_current(self.window)

    def load_texture(self, image_path):
        image = Image.open(image_path)
        image = image.transpose(Image.FLIP_TOP_BOTTOM)
        img_data = np.array(image, dtype=np.uint8)
        texture_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, texture_id)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, image.width, image.height, 0, GL_RGB, GL_UNSIGNED_BYTE, img_data)
        glGenerateMipmap(GL_TEXTURE_2D)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        return texture_id

    def create_shader_program(self):
        vertex_shader = compileShader(vertex_shader_source, GL_VERTEX_SHADER)
        fragment_shader = compileShader(fragment_shader_source, GL_FRAGMENT_SHADER)
        shader_program = compileProgram(vertex_shader, fragment_shader)
        return shader_program

    def create_quad(self):
        self.vertices = np.array([
            -1.0, -1.0, 0.0, 0.0,
             1.0, -1.0, 1.0, 0.0,
             1.0,  1.0, 1.0, 1.0,
            -1.0,  1.0, 0.0, 1.0
        ], dtype=np.float32)
        self.indices = np.array([
            0, 1, 2,
            2, 3, 0
        ], dtype=np.uint32)
        self.VAO = glGenVertexArrays(1)
        self.VBO = glGenBuffers(1)
        self.EBO = glGenBuffers(1)
        glBindVertexArray(self.VAO)
        glBindBuffer(GL_ARRAY_BUFFER, self.VBO)
        glBufferData(GL_ARRAY_BUFFER, self.vertices.nbytes, self.vertices, GL_STATIC_DRAW)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.EBO)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, self.indices.nbytes, self.indices, GL_STATIC_DRAW)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * self.vertices.itemsize, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * self.vertices.itemsize, ctypes.c_void_p(2 * self.vertices.itemsize))
        glEnableVertexAttribArray(1)
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)

    def make_assets(self) -> None:
        self.scene = Scene()
        # self.triangle_mesh = TriangleMesh()
        self.triangle_mesh = Mesh("models/cube.obj", scale=0.5)
        self.triangle_shader = createShader(triangle_vertex_shader_source, triangle_fragment_shader_source)

    def set_onetime_uniforms(self) -> None:
        glUseProgram(self.triangle_shader)
        projection_transform = pyrr.matrix44.create_perspective_projection(
            fovy=45,
            aspect=self.screenWidth / self.screenHeight,
            near=0.1, far=10, dtype=np.float32
        )
        glUniformMatrix4fv(
            glGetUniformLocation(self.triangle_shader, "projection"),
            1, GL_FALSE, projection_transform
        )

    def get_uniform_locations(self) -> None:
        glUseProgram(self.triangle_shader)
        self.modelMatrixLocation = glGetUniformLocation(self.triangle_shader, "model")
        self.viewMatrixLocation = glGetUniformLocation(self.triangle_shader, "view")

    def set_up_input_systems(self) -> None:
        glfw.set_input_mode(self.window, GLFW_CONSTANTS.GLFW_CURSOR, GLFW_CONSTANTS.GLFW_CURSOR_HIDDEN)
        glfw.set_cursor_pos(self.window, self.screenWidth // 2, self.screenHeight // 2)
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
        (w, h) = glfw.get_framebuffer_size(self.window)
        glViewport(0, 0, w, h)
        running = True
        while running:
            if glfw.window_should_close(self.window) or glfw.get_key(self.window, GLFW_CONSTANTS.GLFW_KEY_ESCAPE) == GLFW_CONSTANTS.GLFW_PRESS:
                running = False
            glfw.poll_events()
            self.update_camera_pose()
            self.scene.update(self.frameTime / 16.667)
            glClear(GL_COLOR_BUFFER_BIT)
            glUseProgram(self.shader_program)
            glBindTexture(GL_TEXTURE_2D, self.texture_id)
            glBindVertexArray(self.VAO)
            glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, None)
            glBindVertexArray(0)
            glUseProgram(self.triangle_shader)
            glUniformMatrix4fv(
                self.viewMatrixLocation,
                1, GL_FALSE,
                self.scene.camera.get_view_transform()
            )
            glUniformMatrix4fv(
                self.modelMatrixLocation,
                1, GL_FALSE,
                self.scene.triangle.get_model_transform()
            )
            glBindVertexArray(self.triangle_mesh.vao)
            glDrawArrays(GL_TRIANGLES, 0, self.triangle_mesh.vertex_count)
            glfw.swap_buffers(self.window)
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
        self.currentTime = glfw.get_time()
        delta = self.currentTime - self.lastTime
        if delta >= 1:
            framerate = int(self.numFrames / delta)
            glfw.set_window_title(self.window, f"Running at {framerate} fps.")
            self.lastTime = self.currentTime
            self.numFrames = -1
            self.frameTime = float(1000.0 / max(60, framerate))
        self.numFrames += 1

    def quit(self) -> None:
        self.triangle_mesh.destroy()
        glDeleteProgram(self.shader_program)
        glDeleteProgram(self.triangle_shader)
        glDeleteTextures([self.texture_id])
        glfw.terminate()

class Mesh: 

    def __init__(self, filename, scale=1.0):

        self.scale = scale
        # x, y, z, s, t, nx, ny, nz
        vertices = self.loadMesh(filename)

        self.vertex_count = len(vertices) // 8
        
        # vertices must be in float32 format
        vertices = np.array(vertices, dtype=np.float32)

        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)
        
        # Vertices
        self.vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
        # Position
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 32, ctypes.c_void_p(0))
        # texture
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 32, ctypes.c_void_p(12))

    def loadMesh(self, filename: str):

        v = []
        vt = []
        vn = []

        vertices = []

        with open(filename, "r") as file:

            line = file.readline()

            while line: 
                
                words = line.split(" ")
                if words[0] == "v":
                    v.append(self.read_vertex_data(words))
                    print(self.read_vertex_data(words))
                elif words[0] == "vt":
                    vt.append(self.read_texcoord_data(words))
                elif words[0] == "vn":
                    vn.append(self.read_normal_data(words))
                elif words[0] == "f":
                    self.read_face_data(words, v, vt, vn, vertices)
                line = file.readline()
        
        return vertices
    
    def read_vertex_data(self, words):
        
        return [
            float(words[1])*self.scale, 
            float(words[2])*self.scale,
            float(words[3])*self.scale
        ]

    def read_texcoord_data(self, words):
        
        return [
            float(words[1]), 
            float(words[2])
        ]
    
    def read_normal_data(self, words):
        
        return [
            float(words[1]), 
            float(words[2]),
            float(words[3])
        ]

    def read_face_data(self, words, v, vt, vn, vertices):
        
        triangleCount = len(words) - 3

        for i in range(triangleCount):

            self.make_corner(words[1], v, vt, vn, vertices)
            self.make_corner(words[2 + i], v, vt, vn, vertices)
            self.make_corner(words[3 + i], v, vt, vn, vertices)

    def make_corner(self, corner_description, v, vt, vn, vertices):

        v_vt_vn = corner_description.split("/")
        for element in v[int(v_vt_vn[0])-1]:
            vertices.append(element)
        for element in vt[int(v_vt_vn[1])-1]:
            vertices.append(element)
        for element in vn[int(v_vt_vn[2])-1]:
            vertices.append(element)

    def destroy(self):
        glDeleteVertexArrays(1, (self.vao,))
        glDeleteBuffers(1, (self.vbo,))
        
class TriangleMesh:
    def __init__(self):
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
        glDeleteBuffers(1, (self.vbo,))

myApp = App()
