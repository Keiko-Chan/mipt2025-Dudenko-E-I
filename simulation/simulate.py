from generator import BarCode
import transforms as tr
import markup_tools as markup

from argparse import ArgumentParser
import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter
import cv2
import random
import numpy as np
import json


def nearest_odd(x):
    if x % 2 != 0:
        return x
    else:
        return x - 1


def get_number_of_barcodes():
    numbers = [1, 2, 3, 4, 5, 6, 7, 8] 
    weights = [0.6, 0.2, 0.1, 0.05, 0.025, 0.013, 0.007, 0.005] 
    
    num = random.choices(numbers, weights=weights, k=1)[0]
    print(num)
    return num


def draw_markup_quad(quad, image):
    cv2.line(image, (quad[0][0], quad[0][1]), 
             (quad[1][0], quad[1][1]), (0, 255, 0), thickness=5)
    cv2.line(image, (quad[1][0], quad[1][1]), 
             (quad[2][0], quad[2][1]), (0, 200, 0), thickness=5)
    cv2.line(image, (quad[2][0], quad[2][1]), 
             (quad[3][0], quad[3][1]), (0, 150, 0), thickness=5)
    cv2.line(image, (quad[3][0], quad[3][1]), 
             (quad[0][0], quad[0][1]), (0, 100, 0), thickness=5)
    
    
def add_one_barcode(w_init, h_init, canvas=None, canvas_h=0, canvas_w=0, bar_type="none", data="none"):
    if bar_type == "none":
        bar_type = random.choice(bq_types)
    barcode = BarCode(bar_type, data)
        
    # Rotate barcode
    angle = random.randint(0, 360)
    barcode.rotate(angle)
    
    w_code, h_code = barcode.w, barcode.h
    
    if canvas is None:
        canvas_h = random.randint(min(int(max(w_code, h_code) * 1.5), h_init), max(int(max(w_code, h_code) * 1.5), h_init))
        canvas_w = int(canvas_h * w_init / h_init)
        canvas = np.array(Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255)))
    
    w_key = random.randint(0, canvas_w - w_code - 1)
    h_key = random.randint(0, canvas_h - h_code - 1)
    
    canvas[h_key : h_key + h_code, w_key : w_key + w_code] = np.array(barcode.barcode) 
    barcode.set_key_p(w_key, h_key)
    
    return canvas, canvas_h, canvas_w, barcode


RESULT_PATH = "./result_data"
bq_types = ["qrcode", "azteccode", "pdf417", "datamatrix", "code128", "code39", "ean13", "ean8", "issn", "microqrcode", "upca", "pzn"]


def main(context_path, bar_type="none", amount=1, data="none"):
    
    if not os.path.exists(RESULT_PATH):
        os.makedirs(RESULT_PATH)
    
    if not os.path.exists(RESULT_PATH + "/images"):
        os.makedirs(RESULT_PATH + "/images")
    
    if not os.path.exists(RESULT_PATH + "/markup"):
        os.makedirs(RESULT_PATH + "/markup")

    subfolders = [ f.path for f in os.scandir(context_path) if f.is_dir() ]
    
    folders = [random.choice(subfolders) for _ in range(amount)]
    
    for i in range(0, amount):
        folder = folders[i]
        folder_name = os.path.basename(folder)
        
        inital_image = Path(folder) / "images" / (folder_name + ".tif")

        subsubfolders = [f.path for f in os.scandir(Path(folder) / "images") if f.is_dir()]
        context_folder = random.choice(subsubfolders)
        context_folder_name = os.path.basename(context_folder)
        context_image_path = random.choice([im.path for im in os.scandir(context_folder) if Path(im).suffix == ".tif"])
        context_markup_path = Path(folder) / "ground_truth" / context_folder_name / (Path(os.path.basename(context_image_path)).stem + ".json")
        
        background = Image.open(context_image_path)
        
        w_init, h_init = Image.open(inital_image).size
        
        number_of_barcodes = get_number_of_barcodes()
        barcodes = []
        canvas_h = 0
        canvas_w = 0
        canvas = None
        
        for _ in range(0, int(number_of_barcodes)):
            canvas, canvas_h, canvas_w, barcode = add_one_barcode(w_init, h_init, canvas, 
                                                                  canvas_h, canvas_w, bar_type, data)
            barcodes.append(barcode)
       
        canvas = Image.fromarray(canvas)


        with open(context_markup_path) as f:
            d = json.load(f)
        dst_pts = np.array(d["quad"])
        
        # Projective Transfrom
        pts = [[0, 0], [canvas_w, 0], [canvas_w, canvas_h], [0, canvas_h]]
        pts = np.float32(pts)
        dst_pts = np.float32(dst_pts)
        
        M = cv2.getPerspectiveTransform(pts, dst_pts)
        warped = cv2.warpPerspective(np.array(canvas), M, dsize=background.size)
        
        objects = []
        # MARKUP creation
        for barcode in barcodes:
            code_pts = np.array([[barcode.w_key, barcode.h_key], 
                                 [barcode.w_key, barcode.h_key], 
                                 [barcode.w_key, barcode.h_key], 
                                 [barcode.w_key, barcode.h_key]]) + barcode.pts
            code_pts = np.float32(code_pts)
            code_dst_pts = tr.warp_quad(code_pts, M)
            obj = markup.create_obj_markup(code_dst_pts, 
                                           barcode.bar_type_tag)
            objects.append(obj)
        
        #draw_markup_quad(code_dst_pts, warped)
        warped = Image.fromarray(warped)
        
        mask_im = Image.new("L", background.size, 0)
        draw = ImageDraw.Draw(mask_im)   
        draw.polygon(dst_pts, fill=255)  
        eroded = mask_im.filter(ImageFilter.MinFilter(41)) 
        mask_im = eroded.filter(ImageFilter.GaussianBlur(radius=15))
        
        #smooth_mask = tr.smooth_mask(mask_im, nearest_odd(int(min(w_init, h_init) / 4)))
        smooth_mask = np.stack([mask_im] * 3, axis=-1) / 255
        blended = (smooth_mask * np.array(warped) + np.array(background) * (1 - smooth_mask)).astype(np.uint8)
        blended = Image.fromarray(blended)
        
        #blended = tr.pyramid_blending(background, warped, mask_im, levels=5)
        #blended = (np.stack([mask_im] * 3, axis=-1)/255  * np.array(blended) + np.array(background) * (1 - np.stack([mask_im] * 3, axis=-1)/255)).astype(np.uint8)
        #blended = Image.fromarray(blended)
        
        #background.paste(warped, (0, 0), mask_im) 
        res_markup = markup.create_result_markup(objects, background.size) 
        res_markup = markup.process_imp_det(res_markup)
        
        #blended.save("blended.png")
        #warped.save("warped.png")
        #mask_im.save("mask.png")
        #background.save("background.png")
        
        res_image_path = RESULT_PATH + "/images/" + (str(i) + ".png")
        res_markup_path = RESULT_PATH + "/markup/" + (str(i) + ".png.json")
        
        markup.save_markup(res_markup, res_markup_path)
        blended.save(str(res_image_path))
        
        

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("context_path")
    parser.add_argument('--bar_type', type=str, default="none")
    parser.add_argument('--amount', type=int, default=1)
    parser.add_argument("--data", type=str, default="none")
    
    args = parser.parse_args()

    main(**vars(args))

