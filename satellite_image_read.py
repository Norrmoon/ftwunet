
import numpy as np
import rasterio

file_name = 'C:/Users/norrm/Downloads/sentinel2/2022/20220213_s2b_r036_boa_eov_str26.tif'
width = 186
height = 334

def process_satellite_image(file_name, width, height):
    image = rasterio.open(file_name)
    image_read = image.read()
    image_read_trs = np.transpose(image_read, axes=(0, 2, 1))
    image_padded = np.pad(image_read_trs, ((0,0), (0, width - image_read_trs.shape[1] % width),(0, height - image_read_trs.shape[2] % height)), constant_values=0)
    split_width = np.stack(np.split(image_padded, image_padded.shape[1] // width, axis=1), axis=1)
    split_height = np.concatenate(np.split(split_width, split_width.shape[3] // height, axis=3), axis=1)
    return split_height

image_parts = process_satellite_image(file_name, width, height)

print(f'A kép fel lett bontva {image_parts.shape[1]} részre, {width}x{height} mérettel')
print('image_parts.shape:', image_parts.shape)