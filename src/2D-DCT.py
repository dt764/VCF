'''Exploiting spatial redundancy with the 2D Discrete Cosine Transform of constant block size.'''

import io
#from skimage import io as skimage_io # pip install scikit-image
import numpy as np
#import pywt # pip install pywavelets
import os
import logging
import main
with open("/tmp/description.txt", 'w') as f:
    f.write(__doc__)
import parser
import importlib
import struct

#from DWT import color_dyadic_DWT as DWT
from DCT2D.block_DCT import analyze_image as space_analyze # pip install "DCT2D @ git+https://github.com/vicente-gonzalez-ruiz/DCT2D"
from DCT2D.block_DCT import synthesize_image as space_synthesize
from DCT2D.block_DCT import get_subbands
from DCT2D.block_DCT import get_blocks

from color_transforms.YCoCg import from_RGB # pip install "color_transforms @ git+https://github.com/vicente-gonzalez-ruiz/color_transforms"
from color_transforms.YCoCg import to_RGB

from information_theory import distortion # pip install "information_theory @ git+https://github.com/vicente-gonzalez-ruiz/information_theory"

import cv2

default_block_size = 8
default_CT = "YCoCg"
perceptual_quantization = False
disable_subbands = False

#_parser, parser_encode, parser_decode = parser.create_parser(description=__doc__)

parser.parser_encode.add_argument("-B", "--block_size_DCT", type=parser.int_or_str, help=f"Block size (default: {default_block_size})", default=default_block_size)
parser.parser_encode.add_argument("-t", "--color_transform", type=parser.int_or_str, help=f"Color transform (default: \"{default_CT}\")", default=default_CT)
parser.parser_encode.add_argument("-p", "--perceptual_quantization", action='store_true', help=f"Use perceptual quantization (default: \"{perceptual_quantization}\")", default=perceptual_quantization)
parser.parser_encode.add_argument("-L", "--Lambda", type=parser.int_or_str, help="Relative weight between the rate and the distortion. If provided (float), the block size is RD-optimized between {2**i; i=1,2,3,4,5,6,7}. For example, if Lambda=1.0, then the rate and the distortion have the same weight.")
parser.parser_encode.add_argument("-x", "--disable_subbands", action='store_true', help=f"Disable the coefficients reordering in subbands (default: \"{disable_subbands}\")", default=disable_subbands)

parser.parser_decode.add_argument("-B", "--block_size_DCT", type=parser.int_or_str, help=f"Block size (default: {default_block_size})", default=default_block_size)
parser.parser_decode.add_argument("-t", "--color_transform", type=parser.int_or_str, help=f"Color transform (default: \"{default_CT}\")", default=default_CT)
parser.parser_decode.add_argument("-p", "--perceptual_quantization", action='store_true', help=f"Use perceptual dequantization (default: \"{perceptual_quantization}\")", default=perceptual_quantization)
parser.parser_decode.add_argument("-x", "--disable_subbands", action='store_true', help=f"Disable the coefficients reordering in subbands (default: \"{disable_subbands}\")", default=disable_subbands)

args = parser.parser.parse_known_args()[0]
#try:
#    print("-> Denoising filter =", args.filter) # Don't delete this !
#    denoiser = importlib.import_module(args.filter)
#except AttributeError:
#    # Remember that the filter is only active when decoding, and the import
#    denoiser = importlib.import_module("no_filter")
CT = importlib.import_module(args.color_transform)

class CoDec(CT.CoDec):

    def __init__(self, args):
        logging.debug("trace")
        super().__init__(args)
        self.block_size = args.block_size_DCT
        logging.debug(f"block_size = {self.block_size}")
        if args.perceptual_quantization:
            # See http://www.jatit.org/volumes/Vol70No3/24Vol70No3.pdf
            # Luma
            self.Y_QSSs = np.array([[16, 11, 10, 16, 24, 40, 51, 61], 
                                        [12, 12, 14, 19, 26, 58, 60, 55],
                                        [14, 13, 16, 24, 40, 57, 69, 56],
                                        [14, 17, 22, 29, 51, 87, 80, 62],
                                        [18, 22, 37, 56, 68, 109, 103, 77],
                                        [24, 35, 55, 64, 81, 104, 113, 92],
                                        [49, 64, 78, 87, 103, 121, 120, 101],
                                        [72, 92, 95, 98, 112, 100, 103, 99]])
            # Chroma
            self.C_QSSs = np.array([[17, 18, 24, 47, 99, 99, 99, 99], 
                                        [18, 21, 26, 66, 99, 99, 99, 99],
                                        [24, 26, 56, 99, 99, 99, 99, 99],
                                        [47, 66, 99, 99, 99, 99, 99, 99],
                                        [99, 99, 99, 99, 99, 99, 99, 99],
                                        [99, 99, 99, 99, 99, 99, 99, 99],
                                        [99, 99, 99, 99, 99, 99, 99, 99],
                                        [99, 99, 99, 99, 99, 99, 99, 99]])
            self.C_QSSs = self.C_QSSs.astype(np.uint8)
            self.Y_QSSs = self.Y_QSSs.astype(np.uint8)
            if (self.block_size < 8):
                inter=cv2.INTER_AREA
            else:
                inter=cv2.INTER_LINEAR
            self.C_QSSs = cv2.resize(self.C_QSSs, (self.block_size,self.block_size), interpolation=inter)
            self.Y_QSSs = cv2.resize(self.Y_QSSs, (self.block_size,self.block_size), interpolation=inter)
            #self.quantize_DCT = self.perceptual_quantize_decom
            #self.dequantize_DCT = self.perceptual_dequantize_decom
        else:
            pass
            #self.quantize_DCT = self.quantize_decom
            #self.dequantize_DCT = self.dequantize_decom

        if self.encoding:
            if args.Lambda is not None:
                if not args.perceptual_quantization:
                    self.Lambda = float(args.Lambda)
                    logging.info("optimizing the block size")
                    self.optimize_block_size()
                    logging.info(f"optimal block_size={self.block_size}")
                else:
                    logging.warning("sorry, perceptual quantization is only available for block_size=8")
        if args.quantizer == "deadzone":
            self.offset = 128 # Ojo con esto
        else:
            self.offset = 0

    def UNUSED_oild__pad_and_center_to_multiple_of_block_size(self, array):
        """Pads a 2D NumPy array to the next multiple of a given
        block size in both dimensions, centering the input array in
        the padded array.

        Parameters:
        
        * array (numpy.ndarray): The input 2D array.
        
        * self.block_size (int): The block size (must be a power of 2).

        Returns:
        
        * numpy.ndarray: The padded 2D array with dimensions as
        multiples of the block size.

        """
        
        # Ensure the block size is a power of 2
        if self.block_size & (self.block_size - 1) != 0:
            raise ValueError("Block size must be a power of 2")

        height, width = array.shape

        # Calculate the target dimensions (next multiples of the block size)
        target_height = (height + self.block_size - 1) // self.block_size * self.block_size
        target_width = (width + self.block_size - 1) // self.block_size * self.block_size

        # Calculate padding amounts
        pad_height = target_height - height
        pad_width = target_width - width

        # Distribute the padding equally on both sides (add extra to bottom/right if odd)
        pad_top = pad_height // 2
        pad_bottom = pad_height - pad_top
        pad_left = pad_width // 2
        pad_right = pad_width - pad_left

        # Pad the array with zeros
        padded_array = np.pad(
            array, 
            ((pad_top, pad_bottom), (pad_left, pad_right)), 
            mode='constant', 
            constant_values=0
        )

        return padded_array

    def UNUSED_old__remove_padding(self, padded_array):
        """
        Removes the padding from a padded 2D array, returning the original centered array.

        Parameters:
        
        * padded_array (numpy.ndarray): The padded 2D array.

        Returns:

        * numpy.ndarray: The original array with padding removed.
        """
        original_height, original_width = self.original_shape
        padded_height, padded_width = padded_array.shape

        # Calculate the padding amounts
        pad_height = padded_height - original_height
        pad_width = padded_width - original_width

        # Calculate the slices to extract the original array
        pad_top = pad_height // 2
        pad_left = pad_width // 2

        unpadded_array = padded_array[pad_top:pad_top + original_height, pad_left:pad_left + original_width]

        return unpadded_array

    def pad_and_center_to_multiple_of_block_size(self, img):
        """
        Pads a 3D NumPy array (RGB image) to the next multiple of a given
        block size in both dimensions, centering the input image in the padded image.

        Parameters:
            img (numpy.ndarray): The input 3D image (height x width x channels).

        Returns:
            numpy.ndarray: The padded 3D image with dimensions as multiples of the block size.
        """
        logging.debug("trace")
        if img.ndim != 3:
            raise ValueError("Input image must be a 3D array (height, width, channels).")

        # Save original shape for later use in removing padding
        self.original_shape = img.shape

        height, width, channels = img.shape

        # Calculate the target dimensions (next multiples of the block size)
        target_height = (height + self.block_size - 1) // self.block_size * self.block_size
        target_width = (width + self.block_size - 1) // self.block_size * self.block_size

        # Calculate padding amounts
        pad_height = target_height - height
        pad_width = target_width - width

        # Distribute the padding equally on both sides (add extra to bottom/right if odd)
        pad_top = pad_height // 2
        pad_bottom = pad_height - pad_top
        pad_left = pad_width // 2
        pad_right = pad_width - pad_left

        # Pad the image with zeros (constant value of 0 for RGB)
        padded_img = np.pad(
            img,
            ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),  # No padding for channels
            mode='constant',
            constant_values=0
        )

        return padded_img

    def remove_padding(self, padded_img):
        """
        Removes the padding from a padded 3D image, returning the original centered image.

        Parameters:
            padded_img (numpy.ndarray): The padded 3D image (height x width x channels).

        Returns:
            numpy.ndarray: The original 3D image with padding removed.
        """
        logging.debug("trace")
        if padded_img.ndim != 3:
            raise ValueError("Padded image must be a 3D array (height, width, channels).")

        if self.original_shape is None:
            raise ValueError("Original shape is not set. Pad the image first.")

        original_height, original_width, _  = self.original_shape
        padded_height, padded_width, _ = padded_img.shape

        # Calculate the padding amounts
        pad_height = padded_height - original_height
        pad_width = padded_width - original_width

        # Calculate the slices to extract the original image
        pad_top = pad_height // 2
        pad_left = pad_width // 2

        # Slice to remove padding and recover the original image
        unpadded_img = padded_img[
            pad_top:pad_top + original_height,
            pad_left:pad_left + original_width,
            :
        ]

        return unpadded_img

    def encode_fn(self, in_fn, out_fn):
        logging.debug("trace")
        logging.debug(f"in_fn = {in_fn}")
        logging.debug(f"out_fn = {out_fn}")
        
        #
        # Read the image.
        #
        img = self.encode_read_fn(in_fn).astype(np.float32)

        #
        # Images must have a size multiple of 8.
        #
        self.original_shape = img.shape
        padded_img = self.pad_and_center_to_multiple_of_block_size(img)
        if padded_img.shape != img.shape:
            logging.debug(f"Padding image from dimensions {img.shape} to new dimensions: {padded_img.shape}")
        with open(f"{out_fn}_shape.bin", "wb") as file:
            file.write(struct.pack("iii", *self.original_shape))
        img = padded_img

        #
        # Provides numerical stability to the DCT.
        #
        img -= self.offset
        logging.debug(f"Input to color-DCT with range [{np.min(img)}, {np.max(img)}]")

        #
        # Color transform.
        #
        CT_img = from_RGB(img)

        #
        # Spatial transform (DCT).
        #
        DCT_img = space_analyze(CT_img, self.block_size, self.block_size)

        # Esto no hace falta aquí ##########################
        subband_y_size = int(img.shape[0]/self.block_size)
        subband_x_size = int(img.shape[1]/self.block_size)
        logging.debug(f"subbband_y_size={subband_y_size}, subband_x_size={subband_x_size}")

        #
        # Perceptual quantization.
        #
        if args.perceptual_quantization:
            logging.debug(f"Using perceptual quantization with block_size = {self.block_size}")
            blocks_in_y = int(img.shape[0]/self.block_size)
            blocks_in_x = int(img.shape[1]/self.block_size)
            for by in range(blocks_in_y):
                for bx in range(blocks_in_x):
                    block = DCT_img[by*self.block_size:(by+1)*self.block_size,
                                    bx*self.block_size:(bx+1)*self.block_size,
                                    :]
                    block[..., 0] *= (self.Y_QSSs/121)
                    block[..., 1] *= (self.C_QSSs/99)
                    block[..., 2] *= (self.C_QSSs/99)
                    DCT_img[by*self.block_size:(by+1)*self.block_size,
                            bx*self.block_size:(bx+1)*self.block_size,
                            :] = block

        #
        # Coefficients reordering in subbands. Improves entropy
        # coding.
        #
        if args.disable_subbands:
            decom_img = DCT_img
        else:
            decom_img = get_subbands(DCT_img, self.block_size, self.block_size)

        #
        # Quantization.
        #
        #decom_k = self.quantize_DCT(decom_img)
        #print("----------------->", self.QSS)
        decom_k = self.quantize_decom(decom_img)

        #
        # Make the quantization indexes positive.
        #
        decom_k += self.offset
        logging.debug(f"decom_k[{np.unravel_index(np.argmax(decom_k),decom_k.shape)}]={np.max(decom_k)}")
        logging.debug(f"decom_k[{np.unravel_index(np.argmin(decom_k),decom_k.shape)}]={np.min(decom_k)}")
        if self.args.debug:
            if np.max(decom_k) > 255:
                logging.warning(f"decom_k[{np.unravel_index(np.argmax(decom_k),decom_k.shape)}]={np.max(decom_k)}")
            if np.min(decom_k) < 0:
                logging.warning(f"decom_k[{np.unravel_index(np.argmin(decom_k),decom_k.shape)}]={np.min(decom_k)}")
        #decom_k[0:subband_y_size, 0:subband_x_size, 0] -= 128

        #
        # Compress in memory.
        #
        decom_k = decom_k.astype(np.uint8)
        #print("----------_", decom_k, decom_k.shape)
        #decom_k = np.clip(decom_k, 0, 255).astype(np.uint8)
        decom_k = self.compress(decom_k)

        #
        # Write the code-stream.
        #
        output_size = self.encode_write_fn(decom_k, out_fn)
        #self.BPP = (self.total_output_size*8)/(img.shape[0]*img.shape[1])
        #return rate
        return output_size

    def encode(self, in_fn="/tmp/original.png", out_fn="/tmp/encoded"):
        return self.encode_fn(in_fn, out_fn)

    def decode_fn(self, in_fn, out_fn):
        logging.debug("trace")
        logging.debug(f"in_fn = {in_fn}")
        logging.debug(f"out_fn = {out_fn}")

        #
        # Read the code-stream.
        #
        decom_k = self.decode_read_fn(in_fn)
        with open(f"{in_fn}_shape.bin", "rb") as file:
            self.original_shape = struct.unpack("iii", file.read(12))

        #
        # Decompress the indexes.
        #
        decom_k = self.decompress(decom_k)
        logging.debug(f"original_shape={self.original_shape}, current_shape={decom_k.shape}")

        #
        # Restore original range of the quantization indexes.
        #
        decom_k = decom_k.astype(np.int16)
        #print("----------_", decom_k, decom_k.shape)
        #subband_y_size = int(decom_k.shape[0]/self.block_size)
        #subband_x_size = int(decom_k.shape[1]/self.block_size)
        decom_k -= self.offset

        #
        # Dequantize the indexes to generate the coefficients (by
        # subbands).
        #
        #decom_k[0:subband_y_size, 0:subband_x_size, 0] += 128
        #decom_y = self.dequantize_DCT(decom_k)
        decom_y = self.dequantize_decom(decom_k)
        
        #print(decom_y, decom_y.shape)
        if args.disable_subbands:
            DCT_y = decom_y
        else:
            DCT_y = get_blocks(decom_y, self.block_size, self.block_size)

        #
        # Perceptual de-quantization.
        #
        if args.perceptual_quantization:
            logging.debug(f"Using perceptual de-quantization with block_size = {self.block_size}")
            blocks_in_y = int(DCT_y.shape[0]/self.block_size)
            blocks_in_x = int(DCT_y.shape[1]/self.block_size)
            for by in range(blocks_in_y):
                for bx in range(blocks_in_x):
                    block = DCT_y[by*self.block_size:(by+1)*self.block_size,
                                  bx*self.block_size:(bx+1)*self.block_size,
                                  :].astype(np.float32)
                    block[..., 0] /= (self.Y_QSSs/121)
                    block[..., 1] /= (self.C_QSSs/99)
                    block[..., 2] /= (self.C_QSSs/99)
                    DCT_y[by*self.block_size:(by+1)*self.block_size,
                          bx*self.block_size:(bx+1)*self.block_size,
                          :] = block

        #
        # Inverse spatial transform.
        #
        CT_y = space_synthesize(DCT_y, self.block_size, self.block_size)
        #
        # Restore the original size of the image.
        #
        CT_y = self.remove_padding(CT_y)

        #
        # Restore the RGB domain.
        #
        y = to_RGB(CT_y)

        #
        # Restore the original range of values.
        #
        y += self.offset
        if self.args.debug:
            if np.max(y) > 255:
                logging.warning(f"y[{np.unravel_index(np.argmax(y),y.shape)}]={np.max(y)}")
            if np.min(y) < 0:
                logging.warning(f"y[{np.unravel_index(np.argmin(y),y.shape)}]={np.min(y)}")

        y = CT.CoDec.filter(self, y)

        #
        # Write the image.
        #
        y = np.clip(y, 0, 255).astype(np.uint8)
        output_size = self.decode_write_fn(y, out_fn)
        return output_size

    def decode(self, in_fn="/tmp/encoded", out_fn="/tmp/decoded.png"):
        return self.decode_fn(in_fn, out_fn)

    def quantize_decom(self, decom):
        logging.debug("trace")
        decom_k = self.quantize(decom)
        return decom_k

    def dequantize_decom(self, decom_k):
        logging.debug("trace")
        decom_y = self.dequantize(decom_k)
        return decom_y
    
    def perceptual_quantize_decom(self, decom):
        logging.debug("trace")
        logging.debug(f"Using perceptual quantization with block_size = {self.block_size}")
        subbands_in_y = self.block_size
        subbands_in_x = self.block_size
        subband_y_size = int(decom.shape[0]/self.block_size)
        subband_x_size = int(decom.shape[1]/self.block_size)
        #decom_k = np.empty_like(decom, dtype=np.int16)
        decom_k = decom
        for sb_y in range(subbands_in_y):
            for sb_x in range(subbands_in_x):
                subband = decom[sb_y*subband_y_size:(sb_y+1)*subband_y_size,
                                sb_x*subband_x_size:(sb_x+1)*subband_x_size,
                                :]
                subband_k = np.empty_like(subband, dtype=np.int16)
                self.QSS *= (self.Y_QSSs[sb_y,sb_x]/121)
                subband_k[..., 0] = self.quantize(subband[..., 0])
                self.QSS *= (self.C_QSSs[sb_y,sb_x]/99)
                subband_k[..., 1] = self.quantize(subband[..., 1])
                subband_k[..., 2] = self.quantize(subband[..., 2])
                decom_k[sb_y*subband_y_size:(sb_y+1)*subband_y_size,
                        sb_x*subband_x_size:(sb_x+1)*subband_x_size,
                        :] = subband_k
        return decom_k

    def perceptual_dequantize_decom(self, decom_k):
        logging.debug("trace")
        logging.debug(f"Using perceptual dequantization with block_size = {self.block_size}")
        subbands_in_y = self.block_size
        subbands_in_x = self.block_size
        subband_y_size = int(decom_k.shape[0]/self.block_size)
        subband_x_size = int(decom_k.shape[1]/self.block_size)
        #decom_y = np.empty_like(decom_k, dtype=np.int16)
        decom_y = decom_k
        for sb_y in range(subbands_in_y):
            for sb_x in range(subbands_in_x):
                subband_k = decom_k[sb_y*subband_y_size:(sb_y+1)*subband_y_size,
                                    sb_x*subband_x_size:(sb_x+1)*subband_x_size,
                                    :]
                subband_y = np.empty_like(subband_k, dtype=np.int16)
                self.QSS *= (self.Y_QSSs[sb_y,sb_x]/121)
                subband_y[..., 0] = self.dequantize(subband_k[..., 0])
                self.QSS *= (self.C_QSSs[sb_y,sb_x]/99)
                subband_y[..., 1] = self.dequantize(subband_k[..., 1])
                subband_y[..., 2] = self.dequantize(subband_k[..., 2])
                decom_k[sb_y*subband_y_size:(sb_y+1)*subband_y_size,
                        sb_x*subband_x_size:(sb_x+1)*subband_x_size,
                        :] = subband_y
        return decom_y

    def optimize_block_size(self):
        logging.debug("trace")
        min = 1000000
        img = self.encode_read().astype(np.float32)
        img -= self.offset #np.average(img)
        for block_size in [2**i for i in range(1, 8)]:
            #block_size = 2**i
            CT_img = from_RGB(img)
            DCT_img = space_analyze(CT_img, block_size, block_size)
            decom_img = get_subbands(DCT_img, block_size, block_size)
            '''
            subband_y_size = int(img.shape[0]/block_size)
            subband_x_size = int(img.shape[1]/block_size)
            for sb_y in range(block_size):
                for sb_x in range(block_size):
                    for c in range(3):
                        subband = decom_img[sb_y*subband_y_size:(sb_y+1)*subband_y_size,
                                            sb_x*subband_x_size:(sb_x+1)*subband_x_size,
                                            c]
                        logging.info(f"({sb_y},{sb_x},{c}) subband average = {np.average(subband)}")
 
            logging.info(f"subband_average")
            '''
            decom_k = self.quantize_decom(decom_img)
            decom_k += self.offset
            #decom_k[0:subband_y_size, 0:subband_x_size, 0] -= 128
            if np.max(decom_k) > 255:
                logging.warning(f"decom_k[{np.unravel_index(np.argmax(decom_k),decom_k.shape)}]={np.max(decom_k)}")
            if np.min(decom_k) < 0:
                logging.warning(f"decom_k[{np.unravel_index(np.argmin(decom_k),decom_k.shape)}]={np.min(decom_k)}")
            decom_k_bytes = self.compress(decom_k.astype(np.uint8))
            decom_k_bytes.seek(0)
            rate = len(decom_k_bytes.read())
            decom_k -= self.offset
            #decom_k[0:subband_y_size, 0:subband_x_size, 0] -= 128
            decom_y = self.dequantize_decom(decom_k)
            DCT_y = get_blocks(decom_y, block_size, block_size)
            CT_y = space_synthesize(DCT_y, block_size, block_size)
            y = to_RGB(CT_y)
            y += self.offset
            y = np.clip(y, 0, 255).astype(np.uint8)
            RMSE = distortion.RMSE(img, y)
            J = rate + self.Lambda*RMSE
            logging.debug(f"J={J} for block_size={block_size}")
            if J < min:
                min = J
                self.block_size = block_size

if __name__ == "__main__":
    #parser.description = __doc__
    #parser.parser.description = __doc__
    #parser.description = "Descripción"
    main.main(parser.parser, logging, CoDec)
    #main.main(_parser, logging, CoDec)
