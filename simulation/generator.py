from io import BytesIO
import numpy as np
from PIL import Image

from barcode import generate
from barcode.writer import ImageWriter


gen1_types = ["GS1_128", "UPCA", "ISSN", "ISBN10", "ISBN13", "JAN", "PZN", "Code39", "Code128", "EAN8", "EAN13", "EAN14"]

def gen1(bar_type, data):
    rv = BytesIO()
    generate(bar_type, str(data), writer=ImageWriter(), output=rv)
   
    barcode = Image.open(rv)
    return barcode


gen1("GS1_128", 432)
