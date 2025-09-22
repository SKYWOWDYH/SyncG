import bpy
import math
from mathutils import Vector
from math import radians
import json
import numpy as np
import os
from mathutils import Matrix as mat
from bpy_extras.object_utils import world_to_camera_view
from concurrent.futures import ThreadPoolExecutor, as_completed
import cv2

def load_json(json_file):
    with open(json_file,encoding='UTF-8') as f:
        json_data = json.load(f)
    return json_data

def remove_things_from_collection(collection_name="small_stuff"):
    collection = bpy.data.collections.get(collection_name)
    if collection:
        for obj in collection.objects:
            obj.select_set(True)
        bpy.ops.object.delete()
    else:
        print("Collection not found:", collection_name)

def get_homograph():
    cam_name = "Camera"
    cam = bpy.data.objects[cam_name]
    scene = bpy.context.scene
    vertex_list = []
    cam_vertex_list = []
    v_list = []
    cam_waike_bbox = get_object_bbox('waike',1,1,float=True) # axis follow: top-left (0,0), button-right (1,1)
    cam_waike_x_tl, cam_waike_y_tl = cam_waike_bbox[0:2]
    cam_waike_h = cam_waike_bbox[3]-cam_waike_bbox[1]
    cam_waike_w = cam_waike_bbox[2]-cam_waike_bbox[0]
    new_vertex_list= [Vector((-0.8,0.0,0.0)),Vector((0.0,0.8,0.0)),Vector((0.0,-0.8,0.0)),Vector((0.8,0.0,0.0))]
    for vertex in new_vertex_list:
        x,y,z = vertex
        x_tl, y_tl = (x/2 + 0.5), (0.5 - y/2) # turn to top-left axis
        cam_x, cam_y, cam_z = world_to_camera_view(scene,cam,vertex)
        cam_x_tl, cam_y_tl = cam_x, 1-cam_y

        cam_x_tl_, cam_y_tl_ = (cam_x_tl-cam_waike_x_tl)/cam_waike_w, (cam_y_tl-cam_waike_y_tl)/cam_waike_h # set waike bbox top-left as origin (cropped image)
        vertex_list.append((x_tl,y_tl))
        cam_vertex_list.append((cam_x_tl_, cam_y_tl_))
    vertex_list = np.array(vertex_list)
    cam_vertex_list = np.array(cam_vertex_list)

    h_matrix = cv2.findHomography(cam_vertex_list, vertex_list)[0]
    if h_matrix is None:
        print(cam_vertex_list)
        print(vertex_list)
    return h_matrix


def create_a_circle(modify_degree=0,start_degree=30, end_degree=150,radius=0.93, vertics_num=36,location=(0,0,0),inverted_color=False):
    
    circle_color_material = bpy.data.materials.get('black_color')
    if inverted_color:
        circle_color_material = bpy.data.materials.get('white_color')
    bpy.ops.mesh.primitive_circle_add(vertices=vertics_num, radius=radius,enter_editmode=False, align='WORLD', location=location, scale=(1, 1, 1))
    my_circle = bpy.context.object
    my_circle.data.materials.append(circle_color_material)
    case = np.random.uniform(0,1)
    if case < 0.6:
        my_circle.hide_render = True

    bpy.ops.object.mode_set(mode = 'EDIT') 
    bpy.ops.mesh.select_mode(type="VERT")
    bpy.ops.mesh.select_all(action = 'DESELECT')
    bpy.ops.object.mode_set(mode = 'OBJECT')
    s_degree = 360/vertics_num
    small_modify_degree = 0.5
    for vertex in bpy.context.object.data.vertices:
        x,y,z = vertex.co
        degree_vertex = math.degrees(math.acos(-x/(radius+0.0001)))
        if y<0:
            degree_vertex = 360-degree_vertex
        if (start_degree+modify_degree)<=0 and (end_degree+modify_degree)>=0:
            if (end_degree+modify_degree+s_degree)<=round(degree_vertex,1)<=(360+(start_degree+modify_degree-s_degree)):
                vertex.select = True
        else:
            if not ((start_degree+modify_degree)<=round(degree_vertex,1)<=(end_degree+modify_degree)):
                vertex.select = True
    global homo_matrix
    homo_matrix = get_homograph()
    
    bpy.ops.object.mode_set(mode = 'EDIT') 
    bpy.ops.mesh.delete(type='VERT')
    bpy.ops.mesh.select_all(action = 'DESELECT')


    bpy.ops.object.mode_set(mode = 'OBJECT')
    for vertex in my_circle.data.vertices:
        vertex.select = True
    bpy.ops.object.mode_set(mode = 'EDIT') 
    bpy.ops.mesh.extrude_region_shrink_fatten(TRANSFORM_OT_shrink_fatten={"value":-0.01})
    bpy.ops.object.mode_set(mode='OBJECT')

def move_camera(camera_name,distance_range=[5.1,7], fov_degree_range=[40,80],keep_straight = True):
    distance = np.random.uniform(*distance_range)
    fov_degree = np.random.uniform(*fov_degree_range)
    
    if not keep_straight:
        theta_max = math.radians(30)
        u = np.random.uniform(0, 1)
        cos_theta = 1 - u * (1 - math.cos(theta_max))
        theta = math.acos(cos_theta)
        phi = np.random.uniform(0, 2 * math.pi)
        x = math.sin(theta) * math.cos(phi)
        y = math.sin(theta) * math.sin(phi)
        z = math.cos(theta)
        inital_pos_vec = [x,y,z]
    else:
        inital_pos_vec = [0,0,1]
    norm_ps_vec = inital_pos_vec/np.linalg.norm(inital_pos_vec)
    
    camera = bpy.data.objects.get(camera_name)
    camera.data.angle = math.radians(fov_degree)

    camera.location = (distance * norm_ps_vec[0], distance * norm_ps_vec[1], distance*norm_ps_vec[2])
    
    direction_to_origin = camera.location.normalized()
    rot_quat_origin = direction_to_origin.to_track_quat('Z', 'Y')
    camera.rotation_euler = rot_quat_origin.to_euler()
    if keep_straight:
        camera.rotation_euler[2] = 0

def gen_random_color_material(range_from=0.0, range_to=1.0):
    r = np.random.random()*(range_to-range_from) + range_from
    g = np.random.random()*(range_to-range_from) + range_from
    b = np.random.random()*(range_to-range_from) + range_from
    random_color_material = bpy.data.materials.new(f"{int(r*255)}_{int(g*255)}_{int(b*255)}")
    random_color_material.diffuse_color = (r, g, b, 1)
    return random_color_material

def clean_and_set_material(obj, new_material):
    if len(obj.data.materials) == 0:
        obj.data.materials.append(new_material)
    else:
        obj.data.materials[0] = new_material

def predefined_color():
    color_list = []
    red_color_material = bpy.data.materials.new(name="red")
    red_color_material.diffuse_color = (0.753, 0, 0, 1)

    yellow_color_material = bpy.data.materials.new(name="yellow")
    yellow_color_material.diffuse_color = (1, 1, 0, 1)

    green_color_material = bpy.data.materials.new(name="green")
    green_color_material.diffuse_color = (0, 0.4784, 0.2157, 1)

    blue_color_material = bpy.data.materials.new(name="blue")
    blue_color_material.diffuse_color = (0, 0, 1, 1)

    purple_color_material = bpy.data.materials.new(name="purple")
    purple_color_material.diffuse_color = (0.502, 0, 0.502, 1)

    orange_color_material = bpy.data.materials.new(name="orange")
    orange_color_material.diffuse_color = (1, 0.647, 0, 1)

    pink_color_material = bpy.data.materials.new(name="pink")
    pink_color_material.diffuse_color = (1, 0.753, 0.796, 1)

    color_list = [red_color_material,yellow_color_material,green_color_material,blue_color_material,orange_color_material,pink_color_material]
    
    return color_list

color_list = predefined_color()
color_name_list = [m.name for m in color_list] 

def truncated_normal_rejection(start, end, mean, std): # generate from 0.3-0.8
    while True:
        value = np.random.normal(mean, std)
        if start <= value <= end:
            return value

def create_fakemeter(pointer_rotate_degree = 200, scale_factor = 1.9, text_offset = 0.35, meter_radius = 1, 
                    dial_factor = 0.85, small_stuff_height = 0.001,
                    start_value = 25, small_interval = 3, long_num = 12, 
                    long_interval_degree =25,long_interval_value = 40, text_interval = 2, text_size = 0.15,
                    long_keduxian_len = 0.12, short_keduxian_len = 0.06,is_circle_outside = False, img_output_format='PNG',
                    width=3840,height=2160,use_gpu=[0,1,2,3,4,5,6,7],keep_straight=False, inverted_color=False,point_set=None,
                    img_output_path=None,project_root=None):
    
    remove_things_from_collection(collection_name="small_stuff") # remove the keduxian and text in the small_stuff collection
    empty = bpy.data.objects.get('Empty') # guide the text orientation
    global pointer_name
    choose_which_pointer = np.random.randint(1,4)
    for pointer_index in range(1,4):
        pn = f'pointer{pointer_index}'
        p = bpy.data.objects.get(pn)
        if pointer_index != choose_which_pointer:
            p.hide_render = True
        else:
            p.hide_render = False
    
    # inverse_color = np.random.random() > 0.9 # if true, all scale and text will set to white, pointer will set to white or red, dial will set to black

    pointer_name = f'pointer{choose_which_pointer}'
    pointer = bpy.data.objects.get(pointer_name)
    pointer_material = bpy.data.materials.get('pointer')
    if inverted_color:
        pointer_material = bpy.data.materials.get('white_color')
    if np.random.random() < 0.3: # random generate pointer color
        pointer_material = gen_random_color_material()
    clean_and_set_material(pointer,pointer_material)
    scale_text_material = bpy.data.materials.get('black_color')
    if inverted_color:
        scale_text_material = bpy.data.materials.get('white_color')

    # move text outside keduxian
    text_outside = False
    if np.random.random() < 0.3:
        text_outside = True
        text_offset = long_keduxian_len
        text_outside_and_rotate = False
        if np.random.random() < 0.5:
            text_outside_and_rotate = True

    waike = bpy.data.objects.get('waike')
    # better to assign outside
    waike_materials = ['waike','waike2','waike3','waike4','waike5']
    waike_materials = [bpy.data.materials.get(i) for i in waike_materials]
    move_camera(camera_name='Camera', keep_straight=keep_straight)
    for s in waike.material_slots:
        if s.material in waike_materials:
            s.material = np.random.choice(waike_materials)
            break
    
    # create decal
    decal_value=dict(Scale=(np.random.uniform(0.2,0.8),
                            np.random.uniform(0.2,0.8),
                            1.0),
                    Location=(np.random.uniform(0,5),
                               np.random.uniform(0,5),
                               0))
    # change text
    image_list = ['bar', 'oil', 'pressure', 'sf6gas', 'temperature']
    global image_name
    image_name = np.random.choice(image_list)
    if inverted_color:
        new_image = f'{project_root}/dial_texture/inverted_biaopan_text_{image_name}.png'
    else:
        new_image = f'{project_root}/dial_texture/biaopan_text_{image_name}.png'
    biaopan_image = bpy.data.materials['biaopan_wenzi'].node_tree.nodes['biaopan_image']
    biaopan_image.image = bpy.data.images.load(new_image)
    font_root = f'{project_root}/font_files'
    font_file = np.random.choice(os.listdir(font_root))
    font = bpy.data.fonts.load(os.path.join(font_root, font_file))
    decal_map = bpy.data.materials['biaopan_wenzi'].node_tree.nodes['decal_map']
    for k,v in decal_value.items():
        decal_map.inputs[k].default_value=v 

    decal_color_ramp = bpy.data.materials['biaopan_wenzi'].node_tree.nodes['decal_color_ramp']
    decal_color = (np.random.random(),
                   np.random.random(),
                   np.random.random(),
                   1)
    change_color_ramp(decal_color_ramp,[0., 1.], [(1.,1.,1.,0), decal_color])

    # change glass
    transparent_factor = truncated_normal_rejection(0.3, 0.8, 0.55, 1.0)
    dial_color_ramp = bpy.data.materials['boli_biaopan'].node_tree.nodes['dial_color_ramp']
    change_color_ramp(dial_color_ramp, [transparent_factor, 1.], [(0.,0.,0.,1),(1.,1.,1.,1)])

    modify_degree = (180-long_num*long_interval_degree)/2
    real_pointer_rotate = 180-pointer_rotate_degree-modify_degree
    pointer.rotation_euler = (0,0,math.radians(real_pointer_rotate))

    # the circle outside or inside the keduxian
    if is_circle_outside:
        short_keduxain_ectra_offset = (long_keduxian_len-short_keduxian_len)/2
        circle_r = meter_radius*dial_factor+long_keduxian_len/2
    else:
        short_keduxain_ectra_offset = (-long_keduxian_len+short_keduxian_len)/2
        circle_r = meter_radius*dial_factor-long_keduxian_len/2
    if text_outside:
        circle_r -= text_offset
    global long_scale_mark_kp
    long_scale_mark_kp = []
    for i in range(long_num+1):
        angle = math.radians(i * long_interval_degree+modify_degree)
        if not text_outside:
            bpy.ops.mesh.primitive_cube_add(size=0.2, location=(-meter_radius*dial_factor*math.cos(angle), meter_radius*dial_factor* math.sin(angle), small_stuff_height))
        else:
            bpy.ops.mesh.primitive_cube_add(size=0.2, location=(-(meter_radius*dial_factor-text_offset)*math.cos(angle), (meter_radius*dial_factor-text_offset)* math.sin(angle), small_stuff_height))
        
        bpy.ops.transform.resize(value=(0.06, 0.01, long_keduxian_len/0.2))
        clean_and_set_material(bpy.context.object, scale_text_material)

        direction = Vector((0 + 2 * math.cos(angle), 0 - 2 * math.sin(angle), 0))
        normalized_direction = direction.normalized()

        bpy.context.object.rotation_euler = normalized_direction.to_track_quat('Z', 'Y').to_euler()
        shift = long_keduxian_len/2
        scale_mark_loc = Vector((-(meter_radius*dial_factor-shift)*math.cos(angle), (meter_radius*dial_factor-shift)* math.sin(angle),small_stuff_height))
        camera_loc_kp = world_to_camera(scale_mark_loc,1920,1080)
        long_scale_mark_kp.append(camera_loc_kp)
        
        # how many long keduxian will have a text annotate
        if i%text_interval == 0:
            if not text_outside:
                bpy.ops.object.text_add(location=(-meter_radius * math.cos(angle) + text_offset * math.cos(angle), 
                                               meter_radius * math.sin(angle) - text_offset * math.sin(angle), 
                                               small_stuff_height))
            else:
                bpy.ops.object.text_add(location=(-meter_radius*dial_factor * math.cos(angle), 
                                               meter_radius*dial_factor * math.sin(angle), 
                                               small_stuff_height))
            text = bpy.context.object
            text.data.body = str(start_value+i * long_interval_value)
            text.data.materials.append(scale_text_material)
            
            bpy.ops.transform.resize(value=(text_size, text_size, 0.05))
            if text_outside and text_outside_and_rotate:
                text.rotation_euler[2] = -(angle-math.pi/2)
            bpy.context.object.data.align_x = 'CENTER'
            bpy.context.object.data.align_y = 'CENTER'
            bpy.ops.object.select_all(action='DESELECT')
            bpy.context.view_layer.objects.active = empty
            text.data.font = font
            
        if i != long_num:
            for j in range(1, small_interval):
                angle_short = math.radians(i * long_interval_degree + j * (long_interval_degree/small_interval)+modify_degree)
                if not text_outside:
                    bpy.ops.mesh.primitive_cube_add(size=0.2, location=(-(meter_radius*dial_factor+short_keduxain_ectra_offset)* math.cos(angle_short), 
                                                                     (meter_radius*dial_factor+short_keduxain_ectra_offset)* math.sin(angle_short), 
                                                                     small_stuff_height))
                else:
                    bpy.ops.mesh.primitive_cube_add(size=0.2, location=(-(meter_radius*dial_factor-text_offset+short_keduxain_ectra_offset)* math.cos(angle_short), 
                                                                     (meter_radius*dial_factor-text_offset+short_keduxain_ectra_offset)* math.sin(angle_short), 
                                                                     small_stuff_height))
                bpy.ops.transform.resize(value=(0.05, 0.01, short_keduxian_len/0.2))
                direction_short = Vector((0 + 1.9 * math.cos(angle_short), 0 - 1.9 * math.sin(angle_short), 0))
                normalized_direction_short = direction_short.normalized()
                bpy.context.object.rotation_euler = normalized_direction_short.to_track_quat('Z', 'Y').to_euler()
                bpy.context.object.data.materials.append(scale_text_material)

    create_a_circle(modify_degree=modify_degree,start_degree=0, end_degree=long_interval_degree*long_num, vertics_num=36*4, location=(0,0,small_stuff_height),radius=circle_r,inverted_color=inverted_color)
    
    # make boli waike noise
    bpy.context.scene.render.image_settings.file_format = img_output_format
    bpy.context.scene.render.filepath = img_output_path

    bpy.ops.render.render(write_still=True)


dial_data_parameter =  dict(long_interval_degree =20, # the degree between neighbor long keduxian
                            long_interval_value = 40, # the value between neighbor long keduxian
                            long_num = 12, # the num of long keduxian
                            pointer_rotate_degree = 200, # pointer orientation. take the start long keduxian as 0 degree, and clockwise to add degree
                            start_value = 25, # the value of start long keduxian
                            small_interval = 3, # how many small keduxian will have between 2 long keduxian
                            text_interval = 1,  # decide how frequence the text annotated
                            ) 
                            
dial_model_parameter = dict(long_keduxian_len = 0.12,  # long keduxian length
                            text_size = 0.15, # text size 
                            short_keduxian_len = 0.06, 
                            scale_factor = 1.9, 
                            text_offset = 0.35,
                            meter_radius = 1, 
                            dial_factor = 0.85, 
                            small_stuff_height = 0.001, 
                            is_circle_outside = False,
                            width=3840,
                            height=2160,
                            img_output_format='PNG',
                            img_output_path=None)

def get_text_width_and_height(text_size, text_len=3):
    W = 0.53*text_size*text_len
    H = 0.72*text_size
    return W,H

def change_color_ramp(color_ramp_obj, positions:list, colors:list): # positions = [0.5, 1.0], colors = [(0.,0.,0.,1),(1.,1.,1.,1)]
    for index, e in enumerate(color_ramp_obj.color_ramp.elements):
        e.position = positions[index]
        e.color = colors[index]

def change_glass_bsdf_color(glass_bsdf_obj, color):
    glass_bsdf_obj.inputs['Color'].default_value = color

def get_min_long_interval_degree(text_size, text_offset, meter_radius, text_len=3):
    W,_ = get_text_width_and_height(text_size,text_len)
    r = meter_radius-text_offset
    min_degree = math.degrees(math.asin(W/r))
    n = min_degree//5+1
    min_degree = 5*n
    if min_degree<0:
        print(f"W={W},r={r}")
    return min_degree, n    

def get_min_text_offset(meter_radius,long_keduxian_len,dial_factor,text_size):
    W,H = get_text_width_and_height(text_size)
    small_gap = math.sqrt(W**2+H**2)/2
    min_text_offset = meter_radius*(1-dial_factor)+long_keduxian_len/2+small_gap
    return min_text_offset
    
def get_max_dial_factor(meter_radius, long_keduxian_len):
    max_dial_factor = (meter_radius-long_keduxian_len/2)/meter_radius
    return max_dial_factor

def get_max_long_num(long_interval_degree):
    num = int(360//long_interval_degree)
    max_long_num = num-1
    return max_long_num    

def world_to_camera(world_vector,width,height):
    cam_name = "Camera"
    cam = bpy.data.objects[cam_name]
    scene = bpy.context.scene
    x,y,_ = world_to_camera_view(scene,cam,world_vector)
    voc_x = x
    voc_y = 1-y
    voc_location = [int(voc_x*width),int(voc_y*height)]
    return voc_location

def sort_seg_p(seg_p_list):
    assert len(seg_p_list) > 2
    seg_p_list = sorted(seg_p_list, key=lambda x: x[1],reverse=True)
    start_point = seg_p_list[0]
    # sort seg_point to get polygon
    def polar_angle(point):
        x0,y0 = start_point
        x, y = point
        vector = ((x-x0),(y0-y))
        vector = vector/(np.linalg.norm(vector)+0.0001)
        value = vector[1]
        p = math.degrees(value)
        if vector[1] < 0 and vector[0] > 0:
            p = 360-p
        elif vector[0] < 0:
            p = 180-p
        return p
    sorted_seg_p = [start_point]+sorted(seg_p_list[1:], key=polar_angle)
    
    return sorted_seg_p

def get_cube_seg_points(cam,cube,width,height):
    depsgraph = bpy.context.evaluated_depsgraph_get()
    scene = bpy.context.scene
    obj = cube.evaluated_get(depsgraph)
    mWorld = obj.matrix_world
    vertices = [mWorld @ v.co for v in obj.data.vertices]
    loc_list = []
    for v in vertices:
        x,y,_ = world_to_camera_view(scene,cam,v)
        loc_list.append((x,1-y))
    loc_list = list(set(loc_list))
    loc_list_xmin = min([i[0] for i in loc_list])
    loc_list_xmax = max([i[0] for i in loc_list])
    loc_list_ymin = min([i[1] for i in loc_list])
    loc_list_ymax = max([i[1] for i in loc_list])
    final_result = []
    for i in loc_list:
        if (loc_list_xmin< i[0] < loc_list_xmax) and (loc_list_ymin< i[1] < loc_list_ymax):
            continue
        final_result.append((int(i[0]*width),int(i[1]*height)))
    final_result = sort_seg_p(final_result)
    return final_result

def get_all_cube_seg_points(width,height):
    cam_name = "Camera" #or whatever it is
    cam = bpy.data.objects[cam_name]
    cub_segpoint_list = []
    for obj in bpy.data.objects:
        if obj.type == 'MESH' and obj.name.startswith("Cube"):
            seg_p = get_cube_seg_points(cam,obj,width,height)
            cub_segpoint_list.append(seg_p)
    return cub_segpoint_list

def get_object_bbox(obj_name,width,height,float=False):
    cam_name = "Camera" #or whatever it is
    obj_name = obj_name #or whatever it is
    scene = bpy.context.scene
    cam = bpy.data.objects[cam_name]
    cube = bpy.data.objects[obj_name]

    depsgraph = bpy.context.evaluated_depsgraph_get()
    obj = cube.evaluated_get(depsgraph)
    mWorld = obj.matrix_world
    vertices = [mWorld @ v.co for v in obj.data.vertices]
    xmin,ymin,xmax,ymax = 1,1,0,0
    # location
    for v in vertices:
         x,y,_ = world_to_camera_view(scene,cam,v)
         xmin = min(x,xmin)
         ymin = min(y,ymin)
         xmax = max(x,xmax)
         ymax = max(y,ymax)
    
    if float:
        bbox = [xmin*width,(1-ymax)*height, xmax*width, (1-ymin)*height]
        return bbox
    bbox = [int(xmin*width),int((1-ymax)*height), int(xmax*width), int((1-ymin)*height)] # switch to voc coordinate
    return bbox

def get_object_mask(obj_name,width,height):
    cam_name = "Camera" #or whatever it is
    obj_name = obj_name #or whatever it is
    scene = bpy.context.scene
    cam = bpy.data.objects[cam_name]
    cube = bpy.data.objects[obj_name]

    depsgraph = bpy.context.evaluated_depsgraph_get()
    obj = cube.evaluated_get(depsgraph)
    mWorld = obj.matrix_world
    vertices = [mWorld @ v.co for v in obj.data.vertices]
    xmin,ymin,xmax,ymax = 1,1,0,0
    
    seg_list = []
    # location
    for v in vertices:
         x,y,_ = world_to_camera_view(scene,cam,v)
         seg_list.append((int(x*width),int((1-y)*height))) # switch to voc coordinate
    return seg_list

def get_pointer_location(obj_name,width,height):
    cam_name = "Camera" #or whatever it is
    obj_name = obj_name #or whatever it is
    scene = bpy.context.scene
    cam = bpy.data.objects[cam_name]
    cube = bpy.data.objects[obj_name]
    
    
    depsgraph = bpy.context.evaluated_depsgraph_get()
    obj = cube.evaluated_get(depsgraph)
    mWorld = obj.matrix_world
    
    vertices = [mWorld @ v.co for v in obj.data.vertices]
    max_dist = 0
    # location
    for v in vertices:
         x,y,_ = world_to_camera_view(scene,cam,v)
         y = 1-y
         _dist = (x-0.5)**2+(y-0.5)**2
         if _dist > max_dist:
             max_location = [int(x*width),int(y*height)]
             max_dist = _dist  # more wisely choice is count the location in world first
    return max_location

def get_pointer_bbox(max_location,width,height,bbox_size=0.01):
    x, y = max_location
    voc_bbox = [(x-width*bbox_size),(y-height*bbox_size),(x+width*bbox_size),(y+height*bbox_size)] # better choice is to limit the value in 0-1, but I am lazy
    return voc_bbox

def get_text_bbox(text_obj,width,height):
    cam_name = "Camera" #or whatever it is
    scene = bpy.context.scene
    cam = bpy.data.objects[cam_name]
    cube = text_obj

    depsgraph = bpy.context.evaluated_depsgraph_get()
    obj = cube.evaluated_get(depsgraph)
    mWorld = obj.matrix_world
    vertices = [mWorld @ Vector(v[:]) for v in obj.bound_box]
    xmin,ymin,xmax,ymax = 1,1,0,0
    # location
    for v in vertices:
         x,y,_ = world_to_camera_view(scene,cam,v)
         xmin = min(x,xmin)
         ymin = min(y,ymin)
         xmax = max(x,xmax)
         ymax = max(y,ymax)
    bbox = [int(width*xmin),int((1-ymax)*height), int(width*xmax), int((1-ymin)*height)] # switch to voc coordinate
    return bbox

def get_ann_bbox(width,height):
    ann_list = []
    for obj in bpy.data.objects:
        if obj.type == 'FONT':
            text_ann_dict = dict(type='Text',value=obj.data.body,bbox=get_text_bbox(obj,width,height))
            ann_list.append(text_ann_dict)
    pointer_bbox = get_pointer_bbox(get_pointer_location(pointer_name,width,height),width,height)
    pointer_ann_dict = dict(type='peak',value='peak',bbox=pointer_bbox)
    ann_list.append(pointer_ann_dict)
    return ann_list

def get_ann_seg(width,height):
    mark_mask_dict = dict(type='ScaleMark',mask=get_all_cube_seg_points(width,height))
    pointer_mask_dict = dict(type='pointer',mask=get_object_mask(pointer_name,width,height))
    
    ann_list = [mark_mask_dict,pointer_mask_dict]
    return ann_list

def get_ann_seg_mask(height, width, save_folder, seed):
    pointer = bpy.data.objects.get(pointer_name)
    if not pointer:
        raise ValueError("invaild object name pointer")

    original_visibility = {obj.name: obj.hide_render for obj in bpy.data.objects}

    for obj in bpy.data.objects:
        obj.hide_render = obj.name != pointer_name

    world = bpy.context.scene.world
    if not world:
        world = bpy.data.worlds.new("World")
        bpy.context.scene.world = world
    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links
    bg_node = nodes.get('Background') or nodes.new('ShaderNodeBackground')
    output_node = nodes.get('World Output') or nodes.new('ShaderNodeOutputWorld')
    links.new(bg_node.outputs[0], output_node.inputs[0])
    bg_node.inputs[0].default_value = (0, 0, 0, 1)

    mat_name = "MaskMaterial"
    if mat_name in bpy.data.materials:
        mat = bpy.data.materials[mat_name]
    else:
        mat = bpy.data.materials.new(name=mat_name)
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        nodes.clear()
        emission = nodes.new('ShaderNodeEmission')
        emission.inputs[0].default_value = (1, 1, 1, 1)
        output = nodes.new('ShaderNodeOutputMaterial')
        mat.node_tree.links.new(emission.outputs[0], output.inputs[0])

    if pointer.data.materials:
        pointer.data.materials[0] = mat
    else:
        pointer.data.materials.append(mat)

    render = bpy.context.scene.render
    render.image_settings.file_format = 'PNG'
    render.image_settings.color_mode = 'RGB'
    render.filepath = os.path.join(save_folder, f'sync_{seed}.png')

    if not bpy.context.scene.camera:
        cam = bpy.data.cameras.new("MaskCamera")
        cam_obj = bpy.data.objects.new("MaskCamera", cam)
        bpy.context.scene.collection.objects.link(cam_obj)
        bpy.context.scene.camera = cam_obj

        cam_obj.location = pointer.location + bpy.math.Vector((2, 2, 2))
        cam_obj.rotation_euler = (0.785, 0, 0.785)

    bpy.ops.render.render(write_still=True)
    for obj in bpy.data.objects:
        obj.hide_render = original_visibility.get(obj.name, False)
    

def get_ann_keypoints(width,height):
    circle_keypoints_dict = dict(type='circle', all_kp=long_scale_mark_kp)
    origin_z_shift = [0.04,0.05,0.025]
    origin_point = Vector((0.0,0.0,origin_z_shift[int(pointer_name[-1])-1]))
    pointer_origin_point =  world_to_camera(origin_point,width,height)
    pointer_keypoints_dict = dict(type='pointer',origin_kp=pointer_origin_point, outside_kp=get_pointer_location(pointer_name,width,height))
    keypoint_list = [circle_keypoints_dict,pointer_keypoints_dict]
    return keypoint_list

def get_meter_bbox(width,height):
    return get_object_bbox('waike',width,height)

def get_camera_loc():
    cam_name = "Camera"
    cam = bpy.data.objects[cam_name]
    return list(cam.location)

def rotate_environment_texture(rotation_angle_degrees, axis='X'):
    if not bpy.context.scene.world:
        print("missing world setting for scene")
        return

    world = bpy.context.scene.world
    world.use_nodes = True
    node_tree = world.node_tree
    nodes = node_tree.nodes
    
    env_texture_node = None
    for node in nodes:
        if node.type == 'TEX_ENVIRONMENT':
            env_texture_node = node
            break

    if not env_texture_node:
        print("lost env_texture_node")
        return

    mapping_node = None
    input_socket = env_texture_node.inputs['Vector']
    if input_socket.is_linked:
        linked_node = input_socket.links[0].from_node
        if linked_node.type == 'MAPPING':
            mapping_node = linked_node

    if not mapping_node:
        mapping_node = nodes.new(type='ShaderNodeMapping')
        mapping_node.location = env_texture_node.location
        mapping_node.location.x -= 300
        node_tree.links.new(mapping_node.outputs['Vector'], env_texture_node.inputs['Vector'])
        tex_coord_node = nodes.find("TEX_COORD")
        if tex_coord_node == -1:
            tex_coord_node = nodes.new(type='ShaderNodeTexCoord')
            tex_coord_node.location = mapping_node.location
            tex_coord_node.location.x -= 300
        else:
            tex_coord_node = nodes[tex_coord_node]
            
        node_tree.links.new(tex_coord_node.outputs['Generated'], mapping_node.inputs['Vector'])
    rotation_rad = math.radians(rotation_angle_degrees)
    current_rotation = list(mapping_node.inputs['Rotation'].default_value)

    if axis.upper() == 'X':
        current_rotation[0] = rotation_rad
    elif axis.upper() == 'Y':
        current_rotation[1] = rotation_rad
    elif axis.upper() == 'Z':
        current_rotation[2] = rotation_rad
    else:
        print(f"invalid axis: {axis}. Using 'X','Y', or'Z' instead")
        return
    mapping_node.inputs['Rotation'].default_value = tuple(current_rotation)

def output_json(long_interval_degree,long_interval_value,long_num,
                pointer_rotate_degree,start_value,small_interval,text_interval,seed,
                width,height, point_set,
                save_folder,save_name,scene_number=-1):
    num_keduxian_before = pointer_rotate_degree // long_interval_degree
    min_keduxian = num_keduxian_before*long_interval_value + start_value
    closest_keduxian = min_keduxian
    if (num_keduxian_before*long_interval_degree+long_interval_degree/2)<pointer_rotate_degree:
        closest_keduxian += long_interval_value
    
    ann_meter_bbox = get_meter_bbox(width,height)
    ann_bbox = get_ann_bbox(width,height)
    ann_seg = get_ann_seg(width,height)

    ann_camera_location = get_camera_loc()
    ann_keypoints = get_ann_keypoints(width,height)
    
    ground_truth_value = start_value + pointer_rotate_degree/long_interval_degree*long_interval_value

    output_info_dict = dict(file_name=save_name,
                            long_interval_degree=long_interval_degree,
                            long_interval_value=long_interval_value,
                            long_num=long_num,
                            pointer_rotate_degree=pointer_rotate_degree,
                            start_value=start_value,
                            small_interval=small_interval,
                            text_interval=text_interval,
                            num_keduxian_before=num_keduxian_before,
                            min_keduxian=min_keduxian,
                            closest_keduxian=closest_keduxian,
                            seed=seed,
                            image_name=image_name,
                            scene_number=scene_number,
                            width=width,
                            height=height,
                            point_set=point_set,
                            ground_truth=ground_truth_value,
                            homo_matrix=homo_matrix.tolist(),
                            ann_camera_location=ann_camera_location,
                            meter_bbox_annotations=ann_meter_bbox,
                            bbox_annotations=ann_bbox,
                            seg_annotations=ann_seg,
                            keypoints_annotations=ann_keypoints
                            )

    output_info_json = json.dumps(output_info_dict,indent=2)
    process_json_io(save_folder=save_folder,
                    save_name=save_name,
                    output_info_json=output_info_json)

def process_json_io(save_folder,save_name,output_info_json):
    with open(os.path.join(save_folder,f"{save_name}.json"),'w') as f:
        f.write(output_info_json)

def get_pointer_rotate_degree(long_interval_degree,long_num):
    max_degree = long_interval_degree*long_num
    pointer_rotate_degree = np.random.uniform(0,max_degree)
    return pointer_rotate_degree


def parameter_generate(save_folder,
                       save_name=None,use_gpu=None,width=-1,height=-1):
    dial_data_parameter =  dict(long_interval_degree =20, # the degree between neighbor long keduxian
                                long_interval_value = 40, # the value between neighbor long keduxian
                                long_num = 12, # the num of long keduxian
                                pointer_rotate_degree = 200, # pointer orientation. take the start long keduxian as 0 degree, and clockwise to add degree
                                start_value = 25, # the value of start long keduxian
                                small_interval = 3, # how many small keduxian will have between 2 long keduxian
                                text_interval = 1,  # decide how frequence the text annotated
                                ) 
                                
    dial_model_parameter = dict(long_keduxian_len = 0.12,  # long keduxian length
                                text_size = 0.15, # text size 
                                short_keduxian_len = 0.06, 
                                scale_factor = 1.9, 
                                text_offset = 0.35,
                                meter_radius = 1, 
                                dial_factor = 0.85, 
                                small_stuff_height = 0.002, 
                                is_circle_outside = False,
                                width=width,
                                height=height,
                                use_gpu=use_gpu,
                                keep_straight=True,
                                inverted_color=False,
                                point_set=[],
                                img_output_format='JPEG',
                                img_output_path=None)
    if np.random.random() > 0.8: # 80% change camera
        dial_model_parameter['keep_straight'] = False
        
    dial_data_parameter["start_value"] = int(np.random.uniform(-50,50))
    dial_data_parameter["long_interval_value"] = int(np.random.uniform(1,50)) 
    dial_model_parameter["text_size"] = np.random.uniform(0.06,0.12)
    
    dial_data_parameter["small_interval"] = np.random.randint(2,6)
    dial_model_parameter["long_keduxian_len"] = np.random.uniform(0.10,0.15)

    dial_model_parameter["short_keduxian_len"] = np.random.uniform(0.04,0.07)
    dial_model_parameter["is_circle_outside"] = np.random.randint(0,2)
    
    dial_model_parameter["img_output_path"] = os.path.join(save_folder, f"{save_name}.jpg")
    
    max_dial_factor = get_max_dial_factor(dial_model_parameter["meter_radius"],
                                          dial_model_parameter["long_keduxian_len"])
    dial_model_parameter["dial_factor"] = np.random.uniform((max_dial_factor-0.15),max_dial_factor)
    
    min_text_offset = get_min_text_offset(dial_model_parameter["meter_radius"],
                                          dial_model_parameter["long_keduxian_len"],
                                          dial_model_parameter["dial_factor"],
                                          dial_model_parameter["text_size"])
    dial_model_parameter["text_offset"] = np.random.uniform(min_text_offset-0.01,min_text_offset+0.02)
    
    min_degree, n = get_min_long_interval_degree(dial_model_parameter['text_size'],
                                                 dial_model_parameter["text_offset"],
                                                 dial_model_parameter["meter_radius"])
    dial_data_parameter["long_interval_degree"] = 5*np.random.randint(n, n+2)
    try:
        max_long_num = get_max_long_num(dial_data_parameter["long_interval_degree"])
    except:
        print(min_text_offset)
        print(min_degree)
    dial_data_parameter["long_num"] = np.random.randint(6, max_long_num+1)
    
    dial_data_parameter["text_interval"] = np.random.randint(1,3)
    if dial_data_parameter["text_interval"]==2 and dial_data_parameter["long_num"]%2 == 1:
        dial_data_parameter["long_num"] -=1
         
    dial_data_parameter["pointer_rotate_degree"] = get_pointer_rotate_degree(dial_data_parameter["long_interval_degree"],
                                                                             dial_data_parameter["long_num"])
    
    if np.random.random() < 0.3: # 30% inverted color 
        dial_model_parameter['inverted_color'] = True
    
    return dial_data_parameter, dial_model_parameter

def change_world_environment(hdr_file_path):
    world = bpy.data.worlds.new(name="New World")
    world.use_nodes = True
    nodes = world.node_tree.nodes
    for node in nodes:
        nodes.remove(node)
    env_node = nodes.new(type='ShaderNodeTexEnvironment')
    env_node.location = (0,0)
    env_node.image = bpy.data.images.load(hdr_file_path)
    output_node = nodes.new(type='ShaderNodeOutputWorld')
    output_node.location = (400,0)
    world.node_tree.links.new(env_node.outputs['Color'], output_node.inputs['Surface'])
    for scene in bpy.data.scenes:
        scene.world = world
    
    rotate_environment_texture(90, axis='X')
        
def make_random_sense(scene_root):
    scene_file_list = [i for i in os.listdir(scene_root) if  i.endswith(('hdr','exr')) ]
    scene_file = np.random.choice(scene_file_list)
    scene_file = os.path.join(scene_root, scene_file)
    change_world_environment(scene_file)
    return scene_file

def create_folders(workroot,taskname):
    os.makedirs(os.path.join(workroot, taskname),exist_ok=True)
    os.makedirs(os.path.join(workroot, taskname,'annotations'),exist_ok=True)
    os.makedirs(os.path.join(workroot, taskname,'images'),exist_ok=True)
    os.makedirs(os.path.join(workroot, taskname,'mask_result'),exist_ok=True)



bpy.context.scene.render.engine = 'CYCLES'
bpy.context.preferences.addons["cycles"].preferences.compute_device_type = "CUDA"
bpy.context.scene.cycles.device = "GPU"
bpy.context.preferences.addons["cycles"].preferences.get_devices()

config_path = "<your config path>"
config = load_json(config_path)
use_gpu = config['gpu']
for index,d in enumerate(bpy.context.preferences.addons["cycles"].preferences.devices):
    if index in use_gpu:
        assert d.type != 'CPU'
        d["use"] = True
    else:
        d["use"] = False
bpy.context.scene.cycles.samples = config['num_cycle_samples']

width = config['width']
height= config['height']
bpy.context.scene.render.resolution_x = width
bpy.context.scene.render.resolution_y = height    

project_root = config['project_root']
workroot = f'{project_root}/blender_generate'
scene_root = f'{project_root}/scene_file/'

start_seed = config['start_seed']
end_seed = config['end_seed']

for i in range(start_seed, end_seed+1):
    seed_num = i
    np.random.seed(seed_num)
    scene_file = make_random_sense(scene_root=scene_root)
    sub_folder = config['dataset_name']
    create_folders(workroot, sub_folder)

    dial_data_parameter, dial_model_parameter = parameter_generate(save_folder=f"{workroot}/{sub_folder}/images", 
                                                                save_name=f'sync_{i}',width=width,height=height,
                                                                use_gpu=use_gpu)

    create_fakemeter(**dial_data_parameter,**dial_model_parameter,project_root=project_root)
    output_json(**dial_data_parameter,
                seed=seed_num,width=dial_model_parameter['width'],scene_number=scene_file,
                height=dial_model_parameter['height'],
                point_set=dial_model_parameter['point_set'],
                save_folder=f"{workroot}/{sub_folder}/annotations",
                save_name=f"sync_{i}")
    get_ann_seg_mask(height=height, width=width,save_folder=f"{workroot}/{sub_folder}/masks",seed=seed_num)
