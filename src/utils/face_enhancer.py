import os
import torch 

from gfpgan import GFPGANer

from tqdm import tqdm
from src.utils.npuengine import EngineOV
from src.utils.videoio import load_video_to_cv2
from src.utils.codeformer import setup_model
from src.utils.upscaler import UpscaleModel
import cv2


class GeneratorWithLen(object):
    """ From https://stackoverflow.com/a/7460929 """

    def __init__(self, gen, length):
        self.gen = gen
        self.length = length

    def __len__(self):
        return self.length

    def __iter__(self):
        return self.gen

def enhancer_list(images, method='gfpgan', bg_upsampler='realesrgan'):
    gen = enhancer_generator_no_len(images, method=method, bg_upsampler=bg_upsampler)
    return list(gen)

def enhancer_generator_with_len(images, method='gfpgan', bg_upsampler='realesrgan'):
    """ Provide a generator with a __len__ method so that it can passed to functions that
    call len()"""

    if os.path.isfile(images): # handle video to images
        # TODO: Create a generator version of load_video_to_cv2
        images = load_video_to_cv2(images)

    gen = enhancer_generator_no_len(images, method=method, bg_upsampler=bg_upsampler)
    gen_with_len = GeneratorWithLen(gen, len(images))
    return gen_with_len

def enhancer_generator_no_len(images, method='gfpgan', bg_upsampler='realesrgan'):
    """ Provide a generator function so that all of the enhanced images don't need
    to be stored in memory at the same time. This can save tons of RAM compared to
    the enhancer function. """

    print('face enhancer....')
    if not isinstance(images, list) and os.path.isfile(images): # handle video to images
        images = load_video_to_cv2(images)

    # # ------------------------ set up background upsampler ------------------------
    # if bg_upsampler == 'realesrgan':
    #     bg_upsampler = EngineOV('./bmodel_files/realesrgan-x4_BF16_480.bmodel')
    # else:
    #     bg_upsampler = None
    bg_upsampler = None

    restorer = GFPGANer(
        model_path=model_path,
        upscale=2,
        arch=arch,
        channel_multiplier=channel_multiplier,
        bg_upsampler=bg_upsampler) if method != 'codeformer' else setup_model()
    
    if bg_upsampler is not None:
        bg_upsampler = UpscaleModel()
    # ------------------------ restore ------------------------
    for idx in tqdm(range(len(images)), 'Face Enhancer:'):
        
        if method == 'codeformer':
            img = images[idx]
            r_img = restorer.restore(img) # RGB
        else:
            img = cv2.cvtColor(images[idx], cv2.COLOR_RGB2BGR)
            # restore faces and background if necessary
            cropped_faces, restored_faces, r_img = restorer.enhance(
                img,
                has_aligned=False,
                only_center_face=False,
                paste_back=True)
            r_img = cv2.cvtColor(r_img, cv2.COLOR_BGR2RGB)
        yield r_img
