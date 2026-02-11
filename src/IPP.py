'''IPP coding: block-based motion-compensated video coding with RDO.
Stores residual frames as PNG, motion vectors and frame types as side information.
Applies an external 2D transform codec for each frame residual.'''

import sys
import main
import parser
import entropy_video_coding as EVC
import os
import logging
import numpy as np
from PIL import Image
import av
import importlib
import gzip

with open("/tmp/description.txt", "w") as f:
    f.write(__doc__)

import re
#------------------------------------------------------------
# Argumentos específicos IPP
# ------------------------------------------------------------

parser.parser_encode.add_argument("--input_prefix", required=True, help="Input video")
parser.parser_encode.add_argument("--output_prefix", default="/tmp/ipp", help="Output directory")
parser.parser_encode.add_argument("-G", "--gop_size", type=int, default=10)
parser.parser_encode.add_argument("-b", "--block_size", type=int, default=16, help="IPP block size")
parser.parser_encode.add_argument("-S", "--search_range", type=int, default=4)
parser.parser_encode.add_argument("--lambda_rdo", type=float, default=0.01, help="IPP Lambda for RDO")
parser.parser_encode.add_argument("-T", "--transform", type=str, 
    help=f"2D-transform, default: {EVC.DEFAULT_TRANSFORM}", 
    default=EVC.DEFAULT_TRANSFORM)
parser.parser_encode.add_argument("-N", "--number_of_frames", type=parser.int_or_str, help=f"Number of frames to encode (default: {EVC.N_FRAMES})", default=f"{EVC.N_FRAMES}")

parser.parser_decode.add_argument("-o", "--output_prefix", default="/tmp/ipp", help="Input directory")
parser.parser_decode.add_argument("-G", "--gop_size", type=int, default=10)
parser.parser_decode.add_argument("-b", "--block_size", type=int, default=16, help="IPP block size")
parser.parser_decode.add_argument("-T", "--transform", type=str,
    help=f"2D-transform, default: {EVC.DEFAULT_TRANSFORM}", 
    default=EVC.DEFAULT_TRANSFORM)
parser.parser_decode.add_argument("-N", "--number_of_frames", type=parser.int_or_str, help=f"Number of frames to decode (default: {EVC.N_FRAMES})", default=f"{EVC.N_FRAMES}")

args = parser.parser.parse_known_args()[0]

if __debug__:
    if args.debug:
        print(f"III: Importing {args.transform}")

try:
    transform = importlib.import_module(args.transform)
except ImportError as e:
    print(f"Error: Could not find {args.transform} module ({e})")
    print(f"Make sure '2D-{args.transform}.py' is in the same directory as III.py")
    sys.exit(1)

def is_valid_name(name):
    pattern = r'^encoded_\d{4}\.png$'
    return bool(re.match(pattern, name))

class CoDec(EVC.CoDec):
    def __init__(self, args):
        self.args = args
        self.B = getattr(args, "block_size", None)
        
        try:
            transform_module = importlib.import_module(args.transform)
        except ImportError as e:
            raise ImportError(f"Error: No se encontró el módulo {args.transform} ({e})")
            
        self.transform_codec = transform_module.CoDec(args)
        logging.info(f"Using transform codec: {args.transform}")

    def bye(self):
        pass

    def encode(self):
        os.makedirs(self.args.output_prefix, exist_ok=True)
        container = av.open(self.args.input_prefix)
        B = self.args.block_size
        SEARCH = self.args.search_range
        GOP = self.args.gop_size
        LAMBDA = self.args.lambda_rdo

        ref_frame = None
        frame_idx = 0

        for packet in container.demux():
            for frame in packet.decode():
                curr = np.array(frame.to_image()).astype(np.int16)
                H, W = curr.shape[:2]
                orig_fn = f"{self.args.output_prefix}/original_{frame_idx:04d}.png"
                Image.fromarray(curr.astype(np.uint8)).save(orig_fn)


                # ---------------- Inicialización ----------------
                mv = np.zeros((H // B, W // B, 2), dtype=np.int16)
                modes = np.zeros((H // B, W // B), dtype=np.uint8)

                # ---------------- I-FRAME ----------------
                if ref_frame is None or frame_idx % GOP == 0:
                    #frame_type = "I"
                    # Todo bloque es intra
                    to_save_img = curr.copy()
                    recon = curr.copy()
                else:
                    #frame_type = "P"
                    recon = np.zeros_like(curr)

                    # Prepara imagenes todo I y todo P
                    test_I = curr.copy()  # todo I
                    test_P = np.zeros_like(curr)  # todo P
                    for y in range(0, H, B):
                        for x in range(0, W, B):
                            by, bx = y // B, x // B
                            block = curr[y:y+B, x:x+B]

                            # --- Motion Estimation ---
                            best_sad = np.inf
                            best_pred = None
                            best_dy = best_dx = 0
                            for dy in range(-SEARCH, SEARCH+1):
                                for dx in range(-SEARCH, SEARCH+1):
                                    ry, rx = y+dy, x+dx
                                    if ry<0 or rx<0 or ry+B>H or rx+B>W:
                                        continue
                                    cand = ref_frame[ry:ry+B, x+dx:x+dx+B]
                                    sad = np.sum(np.abs(block - cand))
                                    if sad < best_sad:
                                        best_sad = sad
                                        best_pred = cand
                                        best_dy, best_dx = dy, dx

                            residual = block - best_pred
                            test_P[y:y+B, x:x+B] = residual + 128
                            mv[by, bx] = (best_dy, best_dx)

                    # Guardar imágenes para aplicar transformada
                    test_I_png = f"{self.args.output_prefix}/frame_test_I_{frame_idx:04d}.png"
                    test_P_png = f"{self.args.output_prefix}/frame_test_P_{frame_idx:04d}.png"
                    Image.fromarray(np.clip(test_I, 0, 255).astype(np.uint8)).save(test_I_png)
                    Image.fromarray(np.clip(test_P, 0, 255).astype(np.uint8)).save(test_P_png)
                    self.transform_codec.encode_fn(test_I_png, test_I_png.replace(".png",""))
                    self.transform_codec.encode_fn(test_P_png, test_P_png.replace(".png",""))

                    decoded_I_png = f"{self.args.output_prefix}/frame_test_I_decoded_{frame_idx:04d}.png"
                    decoded_P_png = f"{self.args.output_prefix}/frame_test_P_decoded_{frame_idx:04d}.png"
                    self.transform_codec.decode_fn(test_I_png.replace(".png",""), decoded_I_png)
                    self.transform_codec.decode_fn(test_P_png.replace(".png",""), decoded_P_png)

                    decoded_I = np.array(Image.open(decoded_I_png)).astype(np.int16)
                    decoded_P = np.array(Image.open(decoded_P_png)).astype(np.int16)

                    # Comparación bloque a bloque con RDO simplificado 
                    to_save_img = np.zeros_like(curr)
                    for y in range(0, H, B):
                        for x in range(0, W, B):
                            by, bx = y // B, x // B
                            block_I = decoded_I[y:y+B, x:x+B]
                            block_P = decoded_P[y:y+B, x:x+B]

                            # Distorsión D = MSE
                            D_I = np.mean((curr[y:y+B, x:x+B] - block_I)**2)

                            res_block = decoded_P[y:y+B, x:x+B] - 128
                            dy, dx = mv[by, bx]                        # vector de movimiento
                            pred_block = ref_frame[y+dy:y+dy+B, x+dx:x+dx+B]  # bloque predicho
                            recon_block = pred_block + res_block       # reconstrucción real
                            D_P = np.mean((curr[y:y+B, x:x+B] - recon_block)**2)

                            # Heurística para R
                            R_I = np.sum(np.abs(block_I))
                            R_P = np.sum(np.abs(block_P))

                            # J = D + λ * R
                            J_I = D_I + LAMBDA * R_I
                            J_P = D_P + LAMBDA * R_P

                            if J_I <= J_P:
                                to_save_img[y:y+B, x:x+B] = test_I[y:y+B, x:x+B]
                                modes[by, bx] = 0
                                recon[y:y+B, x:x+B] = test_I[y:y+B, x:x+B]
                            else:
                                to_save_img[y:y+B, x:x+B] = test_P[y:y+B, x:x+B]
                                modes[by, bx] = 1
                                recon[y:y+B, x:x+B] = recon_block

                # Guardar residual final
                residual_png = f"{self.args.output_prefix}/residual_{frame_idx:04d}.png"
                residual_prefix = f"{self.args.output_prefix}/residual_{frame_idx:04d}"
                Image.fromarray(np.clip(to_save_img, 0, 255).astype(np.uint8)).save(residual_png)
                self.transform_codec.encode_fn(residual_png, residual_prefix)

                # Guardar Side Information
                with gzip.GzipFile(f"{self.args.output_prefix}/frame_{frame_idx:04d}_mv.npy.gz", "w") as f:
                    np.save(f, mv)
                with gzip.GzipFile(f"{self.args.output_prefix}/frame_{frame_idx:04d}_modes.npy.gz", "w") as f:
                    np.save(f, modes)

                ref_frame = recon.copy()
                frame_idx += 1
                if self.args.number_of_frames and frame_idx >= self.args.number_of_frames:
                    break

            if self.args.number_of_frames and frame_idx >= self.args.number_of_frames:
                break

        logging.info("IPP encoding finished")

    # --- DECODE ---
    def decode(self):
        B = self.args.block_size
        GOP = self.args.gop_size
        ref_frame = None
        frame_idx = 0

        while True:
            mv_fn = f"{self.args.output_prefix}/frame_{frame_idx:04d}_mv.npy.gz"
            modes_fn = f"{self.args.output_prefix}/frame_{frame_idx:04d}_modes.npy.gz"

            if not os.path.exists(mv_fn) or not os.path.exists(modes_fn):
                break

            # MV
            with gzip.GzipFile(mv_fn, "r") as f:
                mv = np.load(f)

            # MODES
            with gzip.GzipFile(modes_fn, "r") as f:
                modes = np.load(f)

            residual_prefix = f"{self.args.output_prefix}/residual_{frame_idx:04d}"
            residual_decoded_png = f"{self.args.output_prefix}/residual_decoded_{frame_idx:04d}.png"

            self.transform_codec.decode_fn(residual_prefix, residual_decoded_png)
            decoded_img = np.array(Image.open(residual_decoded_png)).astype(np.int16)

            H, W = decoded_img.shape[:2]
            recon = np.zeros_like(decoded_img)

            if ref_frame is None or frame_idx % GOP == 0:
                recon = decoded_img.copy()
            else:
                for y in range(0, H, B):
                    for x in range(0, W, B):
                        by, bx = y // B, x // B
                        mode = modes[by, bx]
                        block_val = decoded_img[y:y+B, x:x+B]

                        if mode == 0:
                            recon[y:y+B, x:x+B] = block_val
                        else:
                            residual = block_val - 128
                            dy, dx = mv[by, bx]
                            pred_block = ref_frame[y+dy:y+dy+B, x+dx:x+dx+B]
                            recon[y:y+B, x:x+B] = pred_block + residual

            recon = np.clip(recon, 0, 255)
            out_fn = f"{self.args.output_prefix}/decoded_{frame_idx:04d}.png"
            Image.fromarray(recon.astype(np.uint8)).save(out_fn)
            logging.info(f"Saved reconstructed frame: {out_fn}")

            ref_frame = recon.copy()
            frame_idx += 1


if __name__ == "__main__":
    main.main(parser.parser, logging, CoDec)
