from io import BytesIO
import numpy as np
from PIL import Image, ImageOps
import cv2
import random
import string

from qrcode.image.styledpil import StyledPilImage
from qrcode.image.styles.moduledrawers.pil import RoundedModuleDrawer
from qrcode.image.styles.colormasks import RadialGradiantColorMask

from barcode import generate
from barcode.writer import ImageWriter

import qrcode
from aztec_code_generator import AztecCode
import treepoem 

import transforms as tr

#https://www.kaggle.com/datasets/kontheeboonmeeprakob/midv500?resource=download
gen1_types = {"GS1_128": "1d", "UPCA": "UPC", "ISSN": "1d", "ISBN10": "1d", "ISBN13": "1d", "JAN": "1d", "PZN": "1d", "Code39": "C39", "Code128": "C128", "EAN8": "ean8", "EAN13": "ean13", "EAN14": "1d"}
gen2_types = {"qrcode": "qr", "azteccode": "az", "pdf417": "pdf", "datamatrix": "dm", "code128": "C128", "code39": "C39", "ean13": "ean13", "ean14": "1d", "ean8": "ean8", "issn": "1d", "microqrcode": "m-qr", "upca": "UPC", "pzn": "1d"}


class BarCode:
    mask: np.ndarray
    barcode: Image
    
    def __init__(self, bar_type, data, size=50):
        self.bar_type = bar_type
        self.rotation = False
        
        if data=="none":
            self.generate_data()
        else:
            self.data = data
        
    
        if bar_type in gen1_types:
            self.gen1()   
            self.bar_type_tag = gen1_types[bar_type]             
        else:
            if bar_type in gen2_types:
                self.gen2()
                self.bar_type_tag = gen2_types[bar_type]  
            else:
                raise ValueError('Unknown bar_type')
        
        if self.bar_type_tag == "az" or     \
           self.bar_type_tag == "qr" or     \
           self.bar_type_tag == "pdf" or    \
           self.bar_type_tag == "dm" or     \
           self.bar_type_tag == "m-qr":
           
           self.dimention = "1d"
        
        else:
           self.dimention = "2d"  
        
        w, h = self.barcode.size
        self.w = w
        self.h = h
        
        self.pts = np.array([[0, self.h], [0, 0], [self.w, 0], [self.w, self.h]])
        #with borders
        self.pts_wb = np.array([[0, self.h], [0, 0], [self.w, 0], [self.w, self.h]])
        #self.barcode = self.barcode.resize((box_size, -1), resample=Image.NEAREST)
    
    def set_key_p(self, w_key, h_key):
        self.w_key = w_key
        self.h_key = h_key        
        
    def resize(self, scale_w=1, scale_h=1, cut=False):
        if cut:
            self.barcode = self.barcode.crop((0, 0, int(self.w / scale_w), int(self.h / scale_h)))
            w, h = self.barcode.size
            self.w = w
            self.h = h
            self.pts_wb = np.array([[0, self.h], [0, 0], [self.w, 0], [self.w, self.h]])
        else:
            self.barcode = self.barcode.resize((int(self.w / scale_w), int(self.h / scale_h)), Image.NEAREST)
            self.w = int(self.w / scale_w)
            self.h = int(self.h / scale_h)
            self.pts = np.array([np.array(self.pts).T[0] / scale_w, np.array(self.pts).T[1] / scale_h]).T
            self.pts_wb = np.array([[0, self.h], [0, 0], [self.w, 0], [self.w, self.h]])
            
    def apply_border(self, border=1):
        if self.rotation:
            print("dont apply borders after roation")
            return
            
        bordered_barcode = ImageOps.expand(self.barcode, 
                                           border=border, 
                                           fill='white')
        self.barcode = bordered_barcode
        w, h = self.barcode.size
        self.w = w
        self.h = h
        self.pts = np.array([[border, h - border], [border, border], 
                             [w - border, border], [w - border, h - border]])
        self.pts_wb = np.array([[0, self.h], [0, 0], [self.w, 0], [self.w, self.h]])

    def gen1(self):
        rv = BytesIO()
        generate(self.bar_type, str(self.data), writer=ImageWriter(), output=rv)
       
        self.barcode = Image.open(rv)
    
    def gen2(self):
        #print(self.data)
        #print(self.bar_type)
        self.barcode = treepoem.generate_barcode(
            barcode_type=str(self.bar_type),  
            data=self.data, 
        )    
        
    def rotate(self, angle):
        R = cv2.getRotationMatrix2D((self.w // 2, self.h // 2), angle, 1)
        R_inv = cv2.getRotationMatrix2D((self.w // 2, self.h // 2), -angle, 1)
        
        #pts = [[0, self.h], [0, 0], [self.w, 0], [self.w, self.h]]
        dst = np.array(tr.rotate_quad(self.pts, R))
        dst_wb = np.array(tr.rotate_quad(self.pts_wb, R))
        
        w_new_wb = int(np.max(dst_wb.T[0]) - np.min(dst_wb.T[0]))
        h_new_wb = int(np.max(dst_wb.T[1]) - np.min(dst_wb.T[1]))
        w_new = int(np.max(dst.T[0]) - np.min(dst.T[0]))
        h_new = int(np.max(dst.T[1]) - np.min(dst.T[1]))

        R[0, 2] += (w_new_wb // 2) - self.w // 2
        R[1, 2] += (h_new_wb // 2) - self.h // 2
        
        self.pts = dst.T
        self.pts[0] += (w_new_wb // 2) - self.w // 2
        self.pts[1] += (h_new_wb // 2) - self.h // 2
        self.pts = self.pts.T
        
        self.pts_wb = dst_wb.T
        self.pts_wb[0] += (w_new_wb // 2) - self.w // 2
        self.pts_wb[1] += (h_new_wb // 2) - self.h // 2
        self.pts_wb = self.pts_wb.T
        
        self.pts = np.array(self.pts).astype(np.int32).tolist()
        self.pts_wb = np.array(self.pts_wb).astype(np.int32).tolist()
        
        img = np.array(self.barcode)
        
        #print(R.shape, img.shape)
        rotated = cv2.warpAffine(img, R, (w_new_wb, h_new_wb), borderValue=(255, 255, 255))
        
        self.barcode = Image.fromarray(rotated)
        self.h = h_new_wb
        self.w = w_new_wb
        self.rotation = True
        
    def generate_data(self):
        match self.bar_type:
            case "qrcode" | "azteccode" | "pdf417":
                length = random.randint(1, 250)
                self.data = ''.join(random.choice(string.digits + 
                                                  string.ascii_letters + 
                                                  string.punctuation) for _ in range(length))
            case "datamatrix":
                length = random.randint(1, 100)
                self.data = ''.join(random.choice(string.digits + 
                                                  string.ascii_letters + 
                                                  string.punctuation) for _ in range(length))
            case "code39" | "code128":
                length = random.randint(1, 25)
                self.data = ''.join(random.choice(string.digits + 
                                                  string.ascii_uppercase) for _ in range(length))
            case "ean13":
                value = random.randint(100000000000, 999999999999)
                if value % 2 == 0:
                    value += 1
                self.data = str(value)
            case "ean8":
                value = random.randint(1000000, 9999999)
                if value % 2 == 0:
                   value += 1
                self.data = str(value)
            case "issn":
                value1 = str(random.randint(1000, 9999))
                value2 = str(random.randint(100, 999))
                self.data = value1 + "-" + value2
            case "microqrcode":
                length = random.randint(1, 15)
                self.data = ''.join(random.choice(string.digits + 
                                                  string.ascii_letters + 
                                                  string.punctuation) for _ in range(length))
            case "upca":
                 self.data = str(random.randint(10000000000, 99999999999))
            case "pzn":
                 self.data = str(random.randint(100000, 900000))
            case _:
                raise NotImplementedError("choose another bar type")

"""
def gen_qr(data):
    #barcode = qrcode.make(data)
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_H,
        box_size=10,
        border=0
    )
    qr.add_data('Some data')
    qr.make(fit=True)

    barcode = qr.make_image()
    
    #img_1 = qr.make_image(image_factory=StyledPilImage, module_drawer=RoundedModuleDrawer())    
    #img_2 = qr.make_image(image_factory=StyledPilImage, color_mask=RadialGradiantColorMask())
    #img_3 = qr.make_image(image_factory=StyledPilImage, embeded_image_path="./test_data/images/codes0.jpg")
    #img_1.save("img_1.png")
    #img_2.save("img_2.png")
    #img_3.save("img_3.png")
    
    return barcode
    
"""
def gen_aztec(data, box_size=50):
    """
    azcode = AztecCode(data)
    matr = np.array(azcode.matrix)
    matr[matr == 0] = 255
    matr[matr == 1] = 0
    barcode = Image.fromarray(matr)
    barcode =  barcode.resize((box_size, box_size), resample=Image.NEAREST)
    """
    barcode = treepoem.generate_barcode(
        barcode_type='azteccode',  
        data=data, 
    ) 
    
    return barcode 


#barcode = gen_1d("GS1_128", 432)
#barcode.save("some_file1.png")

#barcode = gen_qr("GS1_128esxcx")
#barcode.save("some_file2.png")

#barcode = gen_aztec("GS1_128esxcx")
#barcode.save("some_file3.png")

#barcode = gen_datamtrx("GS1_128esxcx")
#barcode.save("some_file4.png")
#barcode = gen_datamtrx("234")
#barcode.save("some_file4.png")
