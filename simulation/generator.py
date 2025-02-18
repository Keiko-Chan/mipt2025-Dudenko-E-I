from io import BytesIO
import numpy as np
from PIL import Image

from barcode import generate
from barcode.writer import ImageWriter

import qrcode
from aztec_code_generator import AztecCode


gen1d_types = ["GS1_128", "UPCA", "ISSN", "ISBN10", "ISBN13", "JAN", "PZN", "Code39", "Code128", "EAN8", "EAN13", "EAN14"]

def gen_1d(bar_type, data):
    rv = BytesIO()
    generate(bar_type, str(data), writer=ImageWriter(), output=rv)
   
    barcode = Image.open(rv)
    return barcode
    

def gen_qr(data, fill_color="black", back_color="white", box_size=25, border=4):
    #barcode = qrcode.make(data)
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=box_size,
        border=border
    )
    qr.add_data('Some data')
    qr.make(fit=True)

    barcode = qr.make_image(fill_color=fill_color, back_color=back_color)
    return barcode
    

def gen_aztec(data, box_size=50):
    azcode = AztecCode(data)
    matr = np.array(azcode.matrix)
    matr[matr == 0] = 255
    matr[matr == 1] = 0
    barcode = Image.fromarray(matr)
    barcode =  barcode.resize((box_size, box_size), resample=Image.NEAREST)
    return barcode


barcode = gen_1d("GS1_128", 432)
barcode.save("some_file1.png")

barcode = gen_qr("GS1_128esxcx")
barcode.save("some_file2.png")

barcode = gen_aztec("GS1_128esxcx")
barcode.save("some_file3.png")
