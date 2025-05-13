import json
import numpy as np
import os.path as osp


def intersect_areas(b1, b2):
    xA = max(b1[0], b2[0])
    yA = max(b1[1], b2[1])
    xB = min(b1[0] + b1[2], b2[0] + b2[2])
    yB = min(b1[1] + b1[3], b2[1] + b2[3])
    # x1, y1, x2, y2
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (b1[2] + 1) * (b1[3] + 1)
    boxBArea = (b2[2] + 1) * (b2[3] + 1)

    return interArea
        

def quad_intersection(quad1, quad2):
    max_x1 = max(quad1[0][0], quad1[1][0], quad1[2][0], quad1[3][0])
    min_x1 = min(quad1[0][0], quad1[1][0], quad1[2][0], quad1[3][0])
    max_y1 = max(quad1[0][1], quad1[1][1], quad1[2][1], quad1[3][1])
    min_y1 = min(quad1[0][1], quad1[1][1], quad1[2][1], quad1[3][1])
    max_x2 = max(quad2[0][0], quad2[1][0], quad2[2][0], quad2[3][0])
    min_x2 = min(quad2[0][0], quad2[1][0], quad2[2][0], quad2[3][0])
    max_y2 = max(quad2[0][1], quad2[1][1], quad2[2][1], quad2[3][1])
    min_y2 = min(quad2[0][1], quad2[1][1], quad2[2][1], quad2[3][1])
    
    b1 = [min_x1, min_y1, max_x1 - min_x1, max_y1 - min_y1]
    b2 = [min_x2, min_y2, max_x2 - min_x2, max_y2 - min_y2]
    
    inter = intersect_areas(b1, b2)
    if inter >= 5:
        return True
    
    return False
    

def create_obj_markup(points, bar_type_tag):
    points = np.array(points).tolist()
    
    if len(points) == 4:
        m_type = "quad"
    elif len(points) > 4:
        m_type = "region"
    else:
        raise ValueError('Number of points < 4')
        
    obj_res = {"data": points, "tags": [bar_type_tag], "type": m_type}
    
    return obj_res


def create_result_markup(objects, im_size):
    im_size = np.array(im_size).tolist()
    res = {"objects": objects, "size": im_size}
    return res
    
    
def process_imp_det(markup):
    for i in range(0, len(markup["objects"])):
        i_points = markup["objects"][i]["data"]
        
        for j in range(i+1, len(markup["objects"])):
            if i==j:
                continue
            j_points = markup["objects"][j]["data"]
            
            if quad_intersection(i_points, j_points) and "id" not in markup["objects"][i]["tags"]:
                markup["objects"][i]["tags"].append("id")
     
    return markup
 

def save_markup(markup, path_to_save):
    with open(path_to_save, 'w') as f:
        out = json.dumps(markup, ensure_ascii=False, indent=4)
        f.write(out)
