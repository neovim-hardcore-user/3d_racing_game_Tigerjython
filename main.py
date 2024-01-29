### general import stuff

from gturtle import *
from math import *
from time import *

### matrix and vector math

def matrix_multiply(mat1, mat2):
    return [[sum(mat1[i][k] * mat2[k][j] for k in range(4)) for j in range(4)] for i in range(4)]

def matrix_vector_multiply(matrix, vector):
    return [sum(row[j] * vector[j] for j in range(len(vector))) for row in matrix]
    
def add(vec1, vec2):
    return [x + y for x, y in zip(vec1, vec2)]

def sub(vec1, vec2):
    return [x - y for x, y in zip(vec1, vec2)]
    
def mul(vec, scalar):
    return [x * scalar for x in vec]
    
def div(vec, scalar):
    return [x / scalar for x in vec]
    
def dot_product(vector1, vector2):
    return sum(x * y for x, y in zip(vector1, vector2))

def cross_product(v1, v2):
    return [v1[1] * v2[2] - v1[2] * v2[1], v1[2] * v2[0] - v1[0] * v2[2], v1[0] * v2[1] - v1[1] * v2[0]]
    
def length(vector):
    return sqrt(sum(x * x for x in vector))    

def normalize(vector):
    l = length(vector)
    return [x / l for x in vector] if l != 0 else vector




def perspective_matrix(aspect_ratio, fov_y, near, far, screen_size):
    tan_half_fov_y = tan(fov_y / 2)
    range_inv = 1 / (far - near)

    return [
        [screen_size / (aspect_ratio * tan_half_fov_y), 0, 0, 0],
        [0, screen_size / tan_half_fov_y, 0, 0],
        [0, 0, -(far + near) * range_inv, -2 * far * near * range_inv],
        [0, 0, -1, 0]
    ]

def translation_matrix(translation_vector):
    return [
        [1, 0, 0, translation_vector[0]],
        [0, 1, 0, translation_vector[1]],
        [0, 0, 1, translation_vector[2]],
        [0, 0, 0, 1]
    ]

def scale_matrix(scale_vector):
    return [
        [scale_vector[0], 0, 0, 0],
        [0, scale_vector[1], 0, 0],
        [0, 0, scale_vector[2], 0],
        [0, 0, 0, 1]
    ]

def rotation_matrix_x(theta):
    return [
        [1, 0, 0, 0],
        [0, cos(theta), -sin(theta), 0],
        [0, sin(theta), cos(theta), 0],
        [0, 0, 0, 1]
    ]

def rotation_matrix_y(theta):
    return [
        [cos(theta), 0, sin(theta), 0],
        [0, 1, 0, 0],
        [-sin(theta), 0, cos(theta), 0],
        [0, 0, 0, 1]
    ]

def rotation_matrix_z(theta):
    return [
        [cos(theta), -sin(theta), 0, 0],
        [sin(theta), cos(theta), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ]
    
def transform_vertices(vertices, tranf_mat):
    tranf_verts = []
    for vertex in vertices:
        vert = matrix_vector_multiply(tranf_mat, vertex + [1])
        tranf_verts.append([vert[v] / vert[3] for v in range(0, 2)] + [vert[2]])
        
    return tranf_verts
    
def is_frontfacing(pos):
    return (pos[2][1] - pos[0][1]) * (pos[1][0] - pos[0][0]) > (pos[1][1] - pos[0][1]) * (pos[2][0] - pos[0][0])


### drawing functions

def draw_triangle(pos):
    setPos(pos[0])
    startPath()
    moveTo(pos[1])
    moveTo(pos[2])
    fillPath()
    
def draw_line(pos1, pos2):
    setPos(pos1)
    moveTo(pos2)

### obj loading

def load_obj(file_path, material_offset):
    vertices = []
    normals = []
    faces = []
    
    material = 0
    materials = []

    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('v '):
                vertex = list(map(float, line.split()[1:]))
                vertices.append(vertex)
            elif line.startswith('usemtl '):
                material = line[7:]
                if material not in materials:
                    materials.append(material)
                    
                material = materials.index(material)
                
            elif line.startswith('f '):
                face = line.split()[1:]
                face = [list(map(int, vertex.split('//'))) for vertex in face]
                face = [face[i][0]-1 for i in range(len(face))] + [face[0][1]-1, material + material_offset]
                faces.append(face)
            elif line.startswith('vn '):
                normal = list(map(float, line.split()[1:]))
                normals.append(normal)

    print(materials)
    return vertices, normals, faces


def load_collision_obj(file_path):
    lines = []
    collision = []
    vertices = []
    normals = []
    with open(file_path, 'r') as file:
        for line in file:
            lines.append(line)
        
            
    for line in lines:
        if line.startswith('v '):
            vertex = list(map(float, line.split()[1:]))
            vertices.append(vertex)
            
        elif line.startswith('vn '):
                normal = list(map(float, line.split()[1:]))
                normals.append(normal)
    
    collision_types = []
    for line in lines:
        if line.startswith('usemtl '):
            collision_type = line[7:]
            if collision_type not in collision_types:
                collision_types.append(collision_type)
                
            collision_type = collision_types.index(collision_type)
        if line.startswith('f '):
            face = line.split()[1:]
            face = [list(map(int, vertex.split('//'))) for vertex in face]
            normal = normals[face[0][1]-1]
            face = [vertices[face[i][0]-1] for i in range(len(face))]
            
            epsilon = 1e-6
            
            edge1 = sub(face[1], face[0])
            edge2 = sub(face[2], face[0])
            
            h = cross_product([0, -1, 0], edge2)
            
            a = dot_product(edge1, h)
            
            
            if a > -epsilon and a < epsilon:
                continue
                
            f = 1.0 / a
            
            collision.append([face[0], f, h, edge1, edge2, normal, collision_type])
            
    return collision
            
                
### 3d drawing and geometry handling


class Ggb:
    def __init__(self):
        self.vertices = []
        self.normals = []
        self.faces = []
        
    def clear(self):
        self.vertices = []
        self.normals = []
        self.faces = []
    
    def add(self, vertices, normals, faces):
        self.faces += [[face[0] + len(self.vertices), 
                        face[1] + len(self.vertices), 
                        face[2] + len(self.vertices), 
                        face[3] + len(self.normals), 
                        face[4]] for face in faces]
                            
        self.vertices += vertices
        self.normals += normals

class Scene:
    def __init__(self):
        makeTurtle()
        
        self.ggb = Ggb()
        self.playground = getPlayground()
        self.playground.enableRepaint(False)
        
    def add_geometry(self, vertices, normals, faces):
        self.ggb.add(vertices, normals, faces)
        
    def clear_geometry(self):
        self.ggb.clear()
    
    def transform_geometry(self, tranf_mat):
        self.ggb.vertices = transform_vertices(self.ggb.vertices, tranf_mat)
    
    def poly_sort(self, poly):
        return -max(self.ggb.vertices[poly[0]][2],  self.ggb.vertices[poly[1]][2],  self.ggb.vertices[poly[2]][2])

    def present(self):
        self.playground.clear()
        
        for face in sorted(self.ggb.faces, key=self.poly_sort):
            pos = [self.ggb.vertices[face[tri]] for tri in range(0, 3)]
            
            if is_frontfacing(pos) and not(min(pos[0][2], pos[1][2], pos[2][2]) <= 0):
                l = max(min(dot_product(self.ggb.normals[face[3]], lightsource), 1), 0) * 1.2 + 0.4
                color = makeColor(min(max(colors[face[4]][0] * l, 0), 1), min(max(colors[face[4]][1] * l, 0), 1), min(max(colors[face[4]][2] * l, 0), 1), colors[face[4]][3])
        
                
                setFillColor(color)
                setPenColor(color)
                draw_triangle(pos)
        self.playground.repaint()

### physics handling


#[face[0], f, h, edge1, edge2

def raycast_y(origin, collision):
    closest_tri = [float('inf')]
    epsilon = 1e-6
    for tri in collision:
        s = sub(origin, tri[0])
        u = tri[1] * dot_product(s, tri[2])
        if u < 0.0 or u > 1.0:
            continue
            
        q = cross_product(s, tri[3])
        v = tri[1] * -q[1]
        
        if v < 0.0 or u + v > 1.0:
            continue
            
        t = tri[1] * dot_product(tri[4], q)
        
        if t > epsilon:
            if t <= closest_tri:
                closest_tri = [t, tri[5], tri[6]]
        
    
    return closest_tri


### input handling

class inputHandler:
    def __init__(self):
        self.mode = 0
        self.keys = [0, 0, 0, 0]
        
        #mode 0
        self.timers = [0, 0, 0, 0]
        
        #mode 1
        
        
        
    def update(self, t):
        key = getKeyCode()
        
        if self.mode == 0:
            if key == 38 or key == 87:
                self.timers[0] = t
            elif key == 40 or key == 83:
                self.timers[1] = t
            elif key == 39 or key == 68:
                self.timers[2] = t
            elif key == 37 or key == 65:
                self.timers[3] = t
                
            for i, k in enumerate(self.keys):
                self.keys[i] = (1 / ((t - self.timers[i]) * 2 + 1)) ** 2
            
        elif self.mode == 1:
            pass
                
        
        

### car

class car:
    def __init__(self, pos):
        self.pos = pos

### geometry

colors = [[0.453, 0.514, 0.022, 1], 
          [0.116, 0.116, 0.116, 1], 
          [0.349, 0.164, 0.048, 1], 
          [0.418, 0.802, 0.0, 1], 
          [0.829, 0.047, 0.020, 0.7], 
          [0.8, 0.8, 0.8, 1], 
          [0.014, 0.802, 0.0, 0.7], 
          [0.139, 0.139, 0.139, 1], 
          [0.316, 0.153, 0.801, 1], 
          [0.131, 0.131, 0.131, 1]]
          


track_vertices, track_normals, track_faces = load_obj("racing.obj", 0)
car_vertices, car_normals, car_faces = load_obj("racingcar.obj", 7)

collision = load_collision_obj("racingcollision.obj")


### mainloop




lightsource = normalize([100, 200, -100])

scene = Scene()

projection_matrix = perspective_matrix(1, radians(90), 0.01, 100, 1000)


lang = pi
ang = -pi/4 
xang = 0
zang = 0
lxang = 0
lzang = 0
angacc = 0
acc = [0, 0, 0]
p = [-20, 15, 0]
lp = p

ih = inputHandler()

#ih.mode = 1

while True:
    scene.clear_geometry()
    t = time()

    scene.add_geometry(track_vertices, track_normals, track_faces)
    
    

    ih.update(t)
        
    

    ang += angacc
    lang = (ang + lang*17) / 18
    angacc *= 0.9

    p[0] += acc[0]
    p[1] += acc[1]
    p[2] += acc[2]
    
    lp = div(add(sub(p, mul(acc, 20)), mul(lp, 20)), 21)
    
    r = raycast_y([p[0], p[1], p[2]], collision)
    
    
        
    angacc -= 0.0015 * ih.keys[2]
        
    angacc += 0.0015 * ih.keys[3]

    

    if r[0] <= 0.5 and r != None:
        p[1] = p[1] - r[0] + 0.5
        
        if r[2] == 0:
            friction = 0.03
        elif r[2] == 1:
            friction = 0.07
        elif r[2] == 2:
            friction = 0.02
            
            
       
        
        acc[0] += sin(ang)*.2 * ih.keys[0] * friction
        acc[2] += cos(ang)*.2 * ih.keys[0] * friction
        
        acc[0] -= sin(ang)*.15 * ih.keys[1] * friction
        acc[2] -= cos(ang)*.15 * ih.keys[1] * friction

    
        acc = sub(acc, mul(r[1], dot_product(acc, r[1])))
        
        acc = sub(acc, mul(acc, friction))
        
        xang = atan2(r[1][2], r[1][1])
        zang = atan2(-r[1][0], r[1][1])
        
        lxang = (xang + lxang * 25)/26
        lzang = (zang + lzang * 25)/26
        
        
    acc[1] -= 0.003

        
    

        
    scene.add_geometry(transform_vertices(car_vertices, matrix_multiply(matrix_multiply(matrix_multiply(translation_matrix([p[0], p[1], p[2]]), rotation_matrix_z(lzang)), rotation_matrix_x(lxang)), rotation_matrix_y(ang))), car_normals, car_faces)
    
    scene.transform_geometry(matrix_multiply(matrix_multiply(matrix_multiply(matrix_multiply(projection_matrix, translation_matrix([0, 0, -20])), rotation_matrix_x(0.4)), rotation_matrix_y(pi-lang)), translation_matrix([-lp[0], -lp[1], -lp[2]])))
    
    scene.present()