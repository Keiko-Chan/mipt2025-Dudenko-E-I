import json
import numpy as np
import os.path as osp


def create_obj_markup(points, bar_type_tag, im_size):
    points = np.array(points).tolist()
    im_size = np.array(im_size).tolist()
    
    if len(points) == 4:
        m_type = "quad"
    elif len(points) > 4:
        m_type = "region"
    else:
        raise ValueError('Number of points < 4')
        
    obj_res = {"data": points, "tags": [bar_type_tag], "type": m_type}
    res = {"objects": [obj_res], "size": im_size}
    
    return res
 

def save_markup(markup, path_to_save):
    with open(path_to_save, 'w') as f:
        out = json.dumps(markup, ensure_ascii=False, indent=4)
        f.write(out)
