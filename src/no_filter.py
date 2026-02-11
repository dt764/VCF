'''Filter's common interface.'''

import logging
import main
with open("/tmp/description.txt", 'w') as f:
    f.write(__doc__)
import importlib
import cv2
import parser
default_filter_size = 5

default_EIC = "TIFF"

# Encoder parser
parser.parser_encode.add_argument("-c", "--entropy_image_codec", help=f"Entropy Image Codec (default: {default_EIC})", default=default_EIC) # Se puede quitar?

# Decoder parser
parser.parser_decode.add_argument("-c", "--entropy_image_codec", help=f"Entropy Image Codec (default: {default_EIC})", default=default_EIC)

args = parser.parser.parse_known_args()[0]
EC = importlib.import_module(args.entropy_image_codec)

class CoDec(EC.CoDec):

    def __init__(self, args):
        logging.debug(f"trace args={args}")
        super().__init__(args)
        logging.debug(f"args = {self.args}")
        self.args = args

    def filter(self, img):
        logging.debug(f"trace y={img}")
        logging.info("no filter")
        return img

if __name__ == "__main__":
    main.main(parser.parser, logging, CoDec)
