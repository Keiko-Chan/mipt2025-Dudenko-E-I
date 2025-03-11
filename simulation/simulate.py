from generator import BarCode
from argparse import ArgumentParser
import os
from pathlib import Path
from PIL import Image, ImageDraw
import cv2
import random
import numpy as np
import json

def find_coeffs(pa, pb):
    matrix = []
    for p1, p2 in zip(pa, pb):
        matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0]*p1[0], -p2[0]*p1[1]])
        matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1]*p1[0], -p2[1]*p1[1]])

    A = np.matrix(matrix, dtype=float)
    B = np.array(pb).reshape(8)

    res = np.dot(np.linalg.inv(A.T * A) * A.T, B)
    return np.array(res).reshape(8)


def main(bar_type, data, context_path, amount=1):
    #
    #barcode.barcode.save("some_file2.png")
    
    subfolders = [ f.path for f in os.scandir(context_path) if f.is_dir() ]
    
    for i in range(0, amount):
        folder = random.choice(subfolders)
        folder_name = os.path.basename(folder)
        
        inital_image = Path(folder) / "images" / (folder_name + ".tif")

        subfolders = [f.path for f in os.scandir(Path(folder) / "images") if f.is_dir()]
        context_folder = random.choice(subfolders)
        context_folder_name = os.path.basename(context_folder)
        context_image_path = random.choice([im.path for im in os.scandir(context_folder) if Path(im).suffix == ".tif"])
        context_markup_path = Path(folder) / "ground_truth" / context_folder_name / (Path(os.path.basename(context_image_path)).stem + ".json")
        
        
        background = Image.open(context_image_path)
        
        w_init, h_init = Image.open(inital_image).size
        
        barcode = BarCode(bar_type, data)
        w_code, h_code = barcode.barcode.size
        
        canvas_h = random.randint(max(w_code, h_code) + max(h_code, w_code), h_init)
        canvas_w = int(canvas_h * w_init / h_init)
       
        canvas = np.array(Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255)))
        
        w_key = random.randint(0, canvas_w - w_code - 1)
        h_key = random.randint(0, canvas_h - h_code - 1)
        
        canvas[h_key : h_key + h_code, w_key : w_key + w_code] = np.array(barcode.barcode) 
        canvas = Image.fromarray(canvas)

        with open(context_markup_path) as f:
            d = json.load(f)
        dst_pts = np.array(d["quad"])
        
        pts = [[0, canvas_h], [canvas_w, canvas_h], [canvas_w, 0], [0, 0]]
        
        print(context_image_path)
        pts = np.float32(pts)
        dst_pts = np.float32(dst_pts)
        
        canvas.save("canvas.png")
      
        M = cv2.getPerspectiveTransform(pts, dst_pts)
        warped = cv2.warpPerspective(np.array(canvas), M, dsize=background.size)
        warped = Image.fromarray(warped)
        
        mask_im = Image.new("L", background.size, 0)
        draw = ImageDraw.Draw(mask_im)   
        draw.polygon(dst_pts, fill=255)        
        
        background.paste(warped, (0, 0), mask_im) 
        
        warped.save("warped.png")
        background.save("background.png")
        


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("bar_type")
    parser.add_argument("data")
    parser.add_argument("context_path")
    parser.add_argument('--amount', type=int, default=1)
    
    args = parser.parse_args()

    main(**vars(args))

