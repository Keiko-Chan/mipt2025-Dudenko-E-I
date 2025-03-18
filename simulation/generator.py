from io import BytesIO
import numpy as np
from PIL import Image

from qrcode.image.styledpil import StyledPilImage
from qrcode.image.styles.moduledrawers.pil import RoundedModuleDrawer
from qrcode.image.styles.colormasks import RadialGradiantColorMask

from barcode import generate
from barcode.writer import ImageWriter

import qrcode
from aztec_code_generator import AztecCode
import treepoem 

#https://www.kaggle.com/datasets/kontheeboonmeeprakob/midv500?resource=download
gen1_types = {"GS1_128": "1d", "UPCA": "UPC", "ISSN": "1d", "ISBN10": "1d", "ISBN13": "1d", "JAN": "1d", "PZN": "1d", "Code39": "C39", "Code128": "C128", "EAN8": "ean8", "EAN13": "ean13", "EAN14": "1d"}
gen2_types = {"qrcode": "qr", "azteccode": "az", "pdf417": "pdf", "datamatrix": "dm", "code128": "C128", "code39": "C39", "ean13": "ean13", "ean14": "1d", "ean8": "ean8", "issn": "1d", "microqrcode": "m-qr", "upca": "UPC", "pzn": "1d"}


class BarCode:
    mask: np.ndarray
    barcode: Image
    
    def __init__(self, bar_type, data, size=50):
        self.bar_type = bar_type
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
        
        #self.barcode = self.barcode.resize((box_size, -1), resample=Image.NEAREST)
    

    def gen1(self):
        rv = BytesIO()
        generate(self.bar_type, str(self.data), writer=ImageWriter(), output=rv)
       
        self.barcode = Image.open(rv)
        
    
    def gen2(self):
        self.barcode = treepoem.generate_barcode(
            barcode_type=str(self.bar_type),  
            data=self.data, 
        )    


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
