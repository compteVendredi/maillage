import numpy as np
import polyscope as ps
import polyscope.imgui as psim


def load_obj_mesh(filepath):
    vertices = []
    faces = []

    with open(filepath, 'r') as file:
        for line in file:
            if line.startswith('v '):

                vertex = list(map(float, line.strip().split()[1:]))
                vertices.append(vertex)
            elif line.startswith('f '):

                face = [int(idx.split('/')[0]) - 1 for idx in line.strip().split()[1:]]
                faces.append(face)
                
    return np.array(vertices, dtype=float), faces


def build_neighbors_dict(vertices, faces):
    neighbors_dict = {}
    for i in range(len(vertices)):
        neighbors_dict[i] = set()
    
    for face in faces:
        for i in range(len(face)):
            for j in range(i+1, len(face)):
                if face[j] not in neighbors_dict[face[i]]:
                    neighbors_dict[face[i]].add(face[j])
                if face[i] not in neighbors_dict[face[j]]:
                    neighbors_dict[face[j]].add(face[i])

    for i in range(len(vertices)):
        if i in neighbors_dict[i]:
            neighbors_dict[i].remove(i)
    return neighbors_dict


def get_neighbors_ring(vertices, ring, neighbors_dict):
    assert ring>=1
    
    old_borders = set()
    borders = set(vertices)
    new_borders = set(vertices)
    
    all_inside = set()

    i=0
    while i<ring:
        old_borders = borders.copy()
        borders = new_borders.copy()
        new_borders = set()
        for j in borders:
            new_borders.update(neighbors_dict[j])
        new_borders -= borders
        new_borders -= old_borders
        i += 1
        
        all_inside.update(borders)
        
    return new_borders, all_inside

def cotangent(angle_radians):
    return 1 / np.tan(angle_radians)

def unit_vector(vector):
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def get_alpha_beta(p, v, neighbors_dict, vertices):
    tmp = neighbors_dict[p].intersection(neighbors_dict[v])
    v_left = vertices[tmp.pop()]
    v_right = vertices[tmp.pop()]
    alpha = angle_between(vertices[p]-v_left, vertices[v]-v_left)
    beta = angle_between(vertices[p]-v_right, vertices[v]-v_right)
    return alpha, beta


def create_old_new(N, between, index_begin, index_end):
    from_old_to_new = dict()
    from_new_to_old = dict()
    
    j=0
    k=len(between)
    l=len(between)+len(index_end)
    for i in range(N):
        if i in between:
            from_old_to_new[i]=j
            from_new_to_old[j]=i
            j+=1
        elif i in index_end:
            from_old_to_new[i]=k
            from_new_to_old[k]=i
            k+=1
        elif i in index_begin:
            from_old_to_new[i]=l
            from_new_to_old[l]=i
            l+=1

    return from_old_to_new,from_new_to_old



####Calcul du laplacian par système linéaire

def calculate_laplacian(index_begin, borders, between, vertices, neighbors_dict, transfer_function):
    index_end = borders
    N = len(between)+len(index_begin)+len(borders)
    A = np.zeros((N,N))
    B = np.zeros((N,))
    laplacian = np.zeros(N)
    
    
    from_old_to_new,from_new_to_old = create_old_new(len(vertices), between,\
    index_begin, index_end)
        
    for i in range(0, len(between)):
        for v in neighbors_dict[from_new_to_old[i]]:
            alpha, beta = get_alpha_beta(from_new_to_old[i]\
            ,v,neighbors_dict,vertices)
            A[i,i] += -(cotangent(alpha) + cotangent(beta))
            A[i,from_old_to_new[v]] = cotangent(alpha) + cotangent(beta)
            
    for i in range(len(between), N):
        A[i,i] = 1
        
    for i in range(len(between)+len(index_end), N):
        B[i] = 1
    

    laplacian = np.linalg.solve(A,B)
    laplacian = transfer_function(laplacian)
    vertices_values = np.zeros((len(vertices),))
    for i in range(len(laplacian)):
        vertices_values[from_new_to_old[i]] = laplacian[i]
    
    return vertices_values


#####Calcul du Laplacian par diffusion + fonction de transfert

def calculate_laplacian_by_diffusion(index_begin, borders, between, vertices, neighbors_dict, transfer_function, num_iterations, alpha):
    index_end = borders
    laplacian = np.zeros(len(vertices))
    for i in index_begin:
        laplacian[i] = 1.0
    for i in index_end:
        laplacian[i] = 0.0
    
    for _ in range(num_iterations):
        new_laplacian = np.zeros(len(vertices))

        for i in range(len(vertices)):
            if not(i in between):
                continue
            somme = 0
            for v in neighbors_dict[i]:
                somme += laplacian[v]
            new_laplacian[i] = alpha*1/len(neighbors_dict[i])*somme+(1-alpha)*laplacian[i]
        for i in between:
            laplacian[i] = new_laplacian[i]
    
    for i in index_begin:
        laplacian[i] = transfer_function(laplacian[i])
    for i in borders:
        laplacian[i] = transfer_function(laplacian[i])
    for i in index_end:
        laplacian[i] = transfer_function(laplacian[i])
        
    return laplacian
    



#######Lissage laplacian + fonction de transfert

def laplacian_smoothing(vertices, alpha, neighbors_dict):
    smooth_vertices = vertices.copy()
    for i in range(len(vertices)):
        if len(neighbors_dict[i])==0:
            continue
        smooth_vertices[i] = (1-alpha)*vertices[i] \
        + (alpha)*np.mean( \
        [vertices[index] for index in neighbors_dict[i]], axis=0)
        
    return smooth_vertices

    
    
########Déformation du maillage par translation pondérée par le Laplacien calculé entre 0 (bord)
# et 1 (poignée) sur le maillage
        
def deform_mesh_laplacian(laplacian_values, vertices, translation_vector):
    deformed_vertices = vertices.copy()
    deformed_vertices += laplacian_values.reshape(len(laplacian_values),1)\
    @ translation_vector.reshape(1,3) 
    return deformed_vertices
    




neighbors_level = [6]
vertices, faces = load_obj_mesh(filepath="bunny.obj")
neighbors_dict = build_neighbors_dict(vertices, faces)
index_begin = set([6699])
center_zone = [6699]
zone_level = [2]
null, index_begin = get_neighbors_ring(set(center_zone), zone_level[0], neighbors_dict)
borders, between = get_neighbors_ring(index_begin, neighbors_level[0], neighbors_dict)
between -= set(index_begin)
transfer_function = {"identity": (lambda x: x),\
 "(x^2 - 1)^2": (lambda x: (x**2-1)**2),\
 "1-x": (lambda x: 1-x)}
transfer_function_selected = ["identity"]
vertices_values = calculate_laplacian(index_begin, borders, between, vertices, neighbors_dict, transfer_function[transfer_function_selected[0]])
laplacian_smoothing_value = [0.5]
translation_vector = np.array([0.0, 0.01, 0.02])


ps.set_program_name("IG2: Projet")

ps.init()
original_mesh = ps.register_surface_mesh("mesh", vertices, faces)
original_mesh.add_scalar_quantity("laplacian_linear_system", vertices_values, enabled=True)
original_mesh.add_scalar_quantity("CONST_laplacian_diffusion",\
calculate_laplacian_by_diffusion(index_begin, borders, between, vertices, neighbors_dict, \
transfer_function[transfer_function_selected[0]], num_iterations=200, alpha=0.25), enabled=False)
smooth_mesh = ps.register_surface_mesh("smooth_mesh", laplacian_smoothing(vertices, 0.5, neighbors_dict), faces, enabled=False)
deform_mesh = ps.register_surface_mesh("deform_mesh", deform_mesh_laplacian(vertices_values, vertices, translation_vector), faces, enabled=False)

parameters = {"smoothing_value": laplacian_smoothing_value[0],\
"translation_vector": translation_vector.copy(), "tranfer_function": transfer_function_selected[0],\
"center_zone": center_zone[0],"neighbors_level": neighbors_level[0],"zone_level": zone_level[0]}


def callback():
    global vertices_values, neighbors_level, index_begin, borders, between, center_zone
    
    psim.SetNextItemOpen(True, psim.ImGuiCond_FirstUseEver)
    if(psim.TreeNode("Mesh actually displayed parameters")):    
        psim.TextUnformatted("smoothing_value: {:.3}".format(parameters["smoothing_value"]))
        psim.TextUnformatted("translation_vector: {}".format(parameters["translation_vector"]))
        psim.TextUnformatted("tranfer_function: {}".format(parameters["tranfer_function"]))
        psim.TextUnformatted("center_zone(index)=1: {}".format(parameters["center_zone"]))
        psim.TextUnformatted("neighbors_level: {}".format(parameters["neighbors_level"]))
        psim.TextUnformatted("zone_level: {}".format(parameters["zone_level"]))
        psim.TreePop()

    psim.SetNextItemOpen(True, psim.ImGuiCond_FirstUseEver)
    if(psim.TreeNode("Zone selection")):
        changed, center_zone[0]\
        = psim.InputInt("center_zone", center_zone[0])
        if center_zone[0]<0:
            center_zone[0]=0
        elif center_zone[0]>=len(vertices):
            center_zone[0]=len(vertices)-1

        changed, zone_level[0] \
        = psim.InputInt("zone_level", zone_level[0])
        if zone_level[0]<1:
            zone_level[0]=1  

        changed, neighbors_level[0] \
        = psim.InputInt("neighbors_level", neighbors_level[0])
        if neighbors_level[0]<1:
            neighbors_level[0]=1
            
            

        if(psim.Button("Apply      ")):
            null, index_begin = get_neighbors_ring(set(center_zone), zone_level[0], neighbors_dict)
            borders, between = get_neighbors_ring(index_begin, neighbors_level[0], neighbors_dict)
            between -= set(index_begin)
            vertices_values = \
            calculate_laplacian(index_begin, borders, between, vertices, neighbors_dict, transfer_function[transfer_function_selected[0]])
            deform_mesh.update_vertex_positions(\
            deform_mesh_laplacian(vertices_values, vertices, translation_vector))
            original_mesh.add_scalar_quantity("laplacian_linear_system", vertices_values, enabled=True)
            parameters["tranfer_function"] = transfer_function_selected[0]
            parameters["translation_vector"] = translation_vector.copy()
            parameters["neighbors_level"] = neighbors_level[0]
            parameters["zone"]=index_begin
            parameters["center_zone"]=center_zone[0]


        psim.TreePop()


    psim.SetNextItemOpen(True, psim.ImGuiCond_FirstUseEver)
    if(psim.TreeNode("Mesh deform translation weighted laplacian")):

        changed, laplacian_smoothing_value[0] \
        = psim.InputFloat("smoothing_value", laplacian_smoothing_value[0])
        if laplacian_smoothing_value[0]<0:
            laplacian_smoothing_value[0]=0.0
        elif laplacian_smoothing_value[0]>1:
            laplacian_smoothing_value[0]=1.0

        if(psim.Button("Apply ")):
            smooth_mesh.update_vertex_positions(laplacian_smoothing(vertices, laplacian_smoothing_value[0], neighbors_dict))
            parameters["smoothing_value"] = laplacian_smoothing_value[0]

        psim.TreePop()
    
    psim.SetNextItemOpen(True, psim.ImGuiCond_FirstUseEver)
    if(psim.TreeNode("Mesh deform translation weighted laplacian")):

        changed, translation_vector[0] \
        = psim.InputFloat("x_translation_vector", translation_vector[0]) 
        changed, translation_vector[1] \
        = psim.InputFloat("y_translation_vector", translation_vector[1]) 
        changed, translation_vector[2] \
        = psim.InputFloat("z_translation_vector", translation_vector[2])

        if(psim.Button("Apply  ")):
            deform_mesh.update_vertex_positions(\
            deform_mesh_laplacian(vertices_values, vertices, translation_vector))
            parameters["translation_vector"] = translation_vector.copy()

        psim.TreePop()


    psim.SetNextItemOpen(True, psim.ImGuiCond_FirstUseEver)
    if(psim.TreeNode("Transfer function")):

        changed = psim.BeginCombo(" ", transfer_function_selected[0])
        if changed:
            for val in transfer_function.keys():
                _, selected = psim.Selectable(val, transfer_function_selected[0]==val)
                if selected:
                    transfer_function_selected[0] = val
            psim.EndCombo()
    
        if(psim.Button("Apply   ")):
            vertices_values = \
            calculate_laplacian(index_begin, borders, between, vertices, neighbors_dict, transfer_function[transfer_function_selected[0]])
            deform_mesh.update_vertex_positions(\
            deform_mesh_laplacian(vertices_values, vertices, translation_vector))
            original_mesh.add_scalar_quantity("laplacian_linear_system", vertices_values, enabled=True)
            parameters["tranfer_function"] = transfer_function_selected[0]
            parameters["translation_vector"] = translation_vector.copy()

        
        psim.TreePop()
 
    

ps.set_user_callback(callback)
ps.show()
ps.clear_user_callback()
