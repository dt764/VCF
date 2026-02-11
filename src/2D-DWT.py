'''Exploiting spatial redundancy with the 2D dyadic Discrete Wavelet Transform.'''

import io
from skimage import io as skimage_io # pip install scikit-image
import numpy as np
import pywt # pip install pywavelets
import os
import logging
import main
with open("/tmp/description.txt", 'w') as f:
    f.write(__doc__)
import parser
import importlib

#from DWT import color_dyadic_DWT as DWT
from DWT2D.color_dyadic_DWT import analyze as space_analyze # pip install "DWT2D @ git+https://github.com/vicente-gonzalez-ruiz/DWT2D"
from DWT2D.color_dyadic_DWT import synthesize as space_synthesize

from color_transforms.YCoCg import from_RGB # pip install "color_transforms @ git+https://github.com/vicente-gonzalez-ruiz/color_transforms"
from color_transforms.YCoCg import to_RGB

default_levels = 5
default_DWT = "db5"
default_CT = "YCoCg"

parser.parser_encode.add_argument("-l", "--levels", type=parser.int_or_str, help=f"Number of decomposition levels (default: {default_levels})", default=default_levels)
parser.parser_encode.add_argument("-w", "--wavelet", type=parser.int_or_str, help=f"Wavelet name (default: \"{default_DWT}\")", default=default_DWT)
parser.parser_encode.add_argument("-t", "--color_transform", type=parser.int_or_str, help=f"Color transform (default: \"{default_CT}\")", default=default_CT)

parser.parser_decode.add_argument("-l", "--levels", type=parser.int_or_str, help=f"Number of decomposition levels (default: {default_levels})", default=default_levels)
parser.parser_decode.add_argument("-w", "--wavelet", type=parser.int_or_str, help=f"Wavelet name (default: \"{default_DWT}\")", default=default_DWT)
parser.parser_decode.add_argument("-t", "--color_transform", type=parser.int_or_str, help=f"Color transform (default: \"{default_CT}\")", default=default_CT)

args = parser.parser.parse_known_args()[0]
CT = importlib.import_module(args.color_transform)

class CoDec(CT.CoDec):

    def __init__(self, args):
        logging.debug("trace")
        super().__init__(args)
        self.levels = args.levels
        logging.info(f"levels = {self.levels}")
        #if self.encoding:
        self.wavelet = pywt.Wavelet(args.wavelet)
        #    with open(f"{args.output}_wavelet_name.txt", "w") as f:
        #        f.write(f"{args.wavelet}")
        #        logging.info(f"Written {args.output}_wavelet_name.txt")
        logging.info(f"wavelet={args.wavelet} ({self.wavelet})")
        #else:
        #    with open(f"{args.input}_wavelet_name.txt", "r") as f:
        #        wavelet_name = f.read()
        #        logging.info(f"Read wavelet = \"{wavelet_name}\" from {args.input}_wavelet_name.txt")
        #        self.wavelet = pywt.Wavelet(wavelet_name)
        #    logging.info(f"wavelet={wavelet_name} ({self.wavelet})")

    def encode_fn(self, in_fn, out_fn):
        logging.debug("trace")
        img = self.encode_read_fn(in_fn).astype(np.int16)
        img_128 = img #- 128 # Reduce the max val in LL
        #print(img_128)
        CT_img = from_RGB(img_128)
        #print(CT_img)
        decom_img = space_analyze(CT_img, self.wavelet, self.levels)
        logging.debug(f"len(decom_img)={len(decom_img)}")
        decom_k = self.quantize_decom_fn(decom_img, out_fn)
        output_size = self.write_decom_fn(decom_k, out_fn)

        #k = self.quantize(CT_img)
        #logging.debug(f"k.shape={k.shape}, k.type={k.dtype}")
        #k[..., 1] += 128
        #k[..., 2] += 128
        #compressed_k = self.compress(k.astype(np.uint8))
        #self.encode_write(compressed_k)

        #self.BPP = (self.total_output_size*8)/(img.shape[0]*img.shape[1])
        #return rate
        return output_size

    def decode_fn(self, in_fn, out_fn):
        logging.debug("trace")
        decom_k = self.read_decom_fn(in_fn)
        decom_y = decom_k
        decom_y = self.dequantize_decom_fn(decom_k, in_fn)
        CT_y = space_synthesize(decom_y, self.wavelet, self.levels)

        #compressed_k = self.decode_read()
        #k = self.decompress(compressed_k).astype(np.int16)
        #logging.debug(f"k.shape={k.shape}, k.type={k.dtype}")
        #k[..., 1] -= 128
        #k[..., 2] -= 128
        #CT_y = self.dequantize(k)
        
        y_128 = to_RGB(CT_y)
        y = y_128# + 128
        y = CT.CoDec.filter(self, y)
        y = np.clip(y, 0, 255).astype(np.uint8)
        output_size = self.decode_write_fn(y, out_fn)
        self.BPP = (self.total_input_size*8)/(y.shape[0]*y.shape[1])
        #return rate
        return output_size

    def encode(self):
        return self.encode_fn(
            in_fn=self.args.original,
            out_fn=self.args.encoded)

    def decode(self):
        return self.decode_fn(
            in_fn=self.args.encoded,
            out_fn=self.args.decoded)

    def quantize_decom_fn(self, decom, fn):
        logging.debug("trace")
        logging.debug(f"trace decom = {decom}")
        logging.debug(f"trace fn = {fn}")
        #print(decom[0])
        fn_subband = f"{fn}_LL_{self.levels}"
        LL_k = super().quantize_fn(decom[0], fn_subband)
        #LL_k[..., 1] += 128
        #LL_k[..., 2] += 128
        decom_k = [LL_k]
        resolution_index = self.levels
        for spatial_resolution in decom[1:]:
            subband_names = ["LH", "HL", "HH"]
            subband_index = 0
            spatial_resolution_k = []
            for subband in spatial_resolution:
                subband_name = subband_names[subband_index]
                fn_subband = f"{fn}_{subband_name}_{resolution_index}"
                subband_k = super().quantize_fn(subband, fn_subband)
                #subband_k += 128
                spatial_resolution_k.append(subband_k)
                subband_index += 1
            decom_k.append(tuple(spatial_resolution_k))
        return decom_k

    def dequantize_decom_fn(self, decom_k, fn):
        logging.debug("trace")
        logging.debug(f"trace decom_k = {decom_k}")
        logging.debug(f"trace fn = {fn}")
        LL_k = decom_k[0]
        #LL_k[..., 1] -= 128
        #LL_k[..., 2] -= 128
        resolution_index = self.levels
        fn_subband = f"{fn}_LL_{self.levels}"
        decom_y = [super().dequantize_fn(LL_k, fn_subband)]
        for spatial_resolution_k in decom_k[1:]:
            subband_names = ["LH", "HL", "HH"]
            subband_index = 0
            spatial_resolution_y = []
            for subband_k in spatial_resolution_k:
                subband_name = subband_names[subband_index]
                #subband_k -= 128
                fn_subband = f"{fn}_{subband_name}_{resolution_index}"
                subband_y = super().dequantize_fn(subband_k, fn_subband)
                spatial_resolution_y.append(subband_y)
                subband_index += 1
            decom_y.append(tuple(spatial_resolution_y))
        return decom_y

    def write_decom_fn(self, decom, fn):
        logging.debug("trace")
        logging.debug(f"decom = {decom}")
        logging.debug(f"trace fn = {fn}")
        LL = decom[0]
        #fn_without_extension = fn.split('.')[0] # Creo que esto se puede quitar ########################################################################################
        #fn_subband = f"{fn_without_extension}_LL_{self.levels}"
        fn_subband = f"{fn}_LL_{self.levels}"
        #LL = io.BytesIO(LL)
        #print(np.max(LL), np.min(LL))
        #LL = self.compress(LL.astype(np.uint8))
        LL += 128
        LL = self.compress(LL.astype(np.uint16))
        #LL = self.compress(LL.astype(np.int16))
        output_size = self.encode_write_fn(LL, fn_subband)
        resolution_index = self.levels
        #aux_decom = [decom[0][..., 0]] # Used for computing slices
        for spatial_resolution in decom[1:]:
            subband_names = ["LH", "HL", "HH"]
            subband_index = 0
            #aux_resol = [] # Used for computing slices
            for subband_name in subband_names:
                #fn_subband = f"{fn_without_extension}_{subband_name}_{resolution_index}"
                fn_subband = f"{fn}_{subband_name}_{resolution_index}"
                #SP = io.BytesIO(spatial_resolution[subband_index])
                subband = spatial_resolution[subband_index]
                subband += 128
                SP = self.compress(subband.astype(np.uint8))
                #SP = self.compress_fn(spatial_resolution[subband_index].astype(np.uint8), fn)
                #SP = self.compress(spatial_resolution[subband_index].astype(np.uint16))
                #SP = self.compress(spatial_resolution[subband_index].astype(np.int16))
                output_size += self.encode_write_fn(SP, fn_subband)
                #aux_resol.append(spatial_resolution[subband_index][..., 0])
                subband_index += 1
            resolution_index -= 1
            #aux_decom.append(tuple(aux_resol))
        #self.slices = pywt.coeffs_to_array(aux_decom)[1]
        #return slices
        return output_size

    def read_decom_fn(self, fn):
        logging.debug("trace")
        logging.debug(f"fn = {fn}")
        #fn_without_extension = fn.split('.')[0] #############################3
        #fn_subband = f"{fn_without_extension}_LL_{self.levels}"
        fn_subband = f"{fn}_LL_{self.levels}"
        LL = self.decode_read_fn(fn_subband)
        #LL = self.decompress_fn(LL, fn).astype(np.int16)
        LL = self.decompress(LL).astype(np.int16)
        LL -= 128
        #LL = self.decompress(LL).astype(np.uint16)
        decom = [LL]
        resolution_index = self.levels
        for l in range(self.levels, 0, -1):
            subband_names = ["LH", "HL", "HH"]
            spatial_resolution = []
            for subband_name in subband_names:
                #fn_subband = f"{fn_without_extension}_{subband_name}_{resolution_index}"
                fn_subband = f"{fn}_{subband_name}_{resolution_index}"
                subband = self.decode_read_fn(fn_subband)
                #subband = self.decompress_fn(subband, fn).astype(np.int16)
                subband = self.decompress(subband).astype(np.int16)
                subband -= 128
                spatial_resolution.append(subband)
            decom.append(tuple(spatial_resolution))
            resolution_index -= 1
        return decom

    def UNUSED_quantize_decom(self, decom):
        logging.debug("trace")
        decom_k = [self.quantize(decom[0])] # LL subband
        for spatial_resolution in decom[1:]:
            spatial_resolution_k = []
            for subband in spatial_resolution:
                subband_k = self.quantize(subband)
                spatial_resolution_k.append(subband_k)
            decom_k.append(tuple(spatial_resolution_k))
        return decom_k

    def UNUSED_dequantize_decom(self, decom_k):
        logging.debug("trace")
        decom_y = [self.dequantize(decom_k[0])]
        for spatial_resolution_k in decom_k[1:]:
            spatial_resolution_y = []
            for subband_k in spatial_resolution_k:
                subband_y = self.dequantize(subband_k)
                spatial_resolution_y.append(subband_y)
            decom_y.append(tuple(spatial_resolution_y))
        return decom_y

    def UNUSED_quantize(self, subband):
        '''Quantize the image.'''
        logging.debug("trace")
        #k = self.Q.encode(subband)
        #k = super().quantize(subband)
        k = subband
        k += 32768
        k = k.astype(np.uint16)
        logging.debug(f"k.shape={k.shape} k.dtype={k.dtype}")
        return k

    def UNUSED_dequantize(self, k):
        '''"Dequantize" an image.'''
        logging.debug("trace")
        k = k.astype(np.int16)
        k -= 32768
        #self.Q = Quantizer(Q_step=QSS, min_val=min_index_val, max_val=max_index_val)
        #y = self.Q.decode(k)
        #y = super().dequantize(k)
        y = k
        logging.debug(f"y.shape={y.shape} y.dtype={y.dtype}")
        return y

    '''
    def __save_fn(self, img, fn):
        io.imsave(fn, img, check_contrast=False)
        self.required_bytes = os.path.getsize(fn)
        logging.info(f"Written {self.required_bytes} bytes in {fn}")

    def __read_fn(self, fn):
        img = io.imread(fn)
        logging.info(f"Read {fn} of shape {img.shape}")
        return img
    '''

if __name__ == "__main__":
    main.main(parser.parser, logging, CoDec)
