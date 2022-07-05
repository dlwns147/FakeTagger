import numpy
import cv2
import os
from umeyama import umeyama

def get_image_paths(directory):
    return [x.path for x in os.scandir(directory) if x.name.endswith(".jpg") or x.name.endswith(".png")]

def load_images(image_paths, convert=None):
    iter_all_images = (cv2.resize(cv2.imread(fn), (256, 256)) for fn in image_paths)
    if convert:
        iter_all_images = (convert(img) for img in iter_all_images)
    for i, image in enumerate(iter_all_images):
        if i == 0:
            all_images = numpy.empty((len(image_paths),) + image.shape, dtype=image.dtype)
        all_images[i] = image
    return all_images

def random_warp(image):
    assert image.shape == (256, 256, 3)
    range_ = numpy.linspace(128 - 80, 128 + 80, 5)
    mapx = numpy.broadcast_to(range_, (5, 5))
    mapy = mapx.T

    mapx = mapx + numpy.random.normal(size=(5, 5), scale=5)
    mapy = mapy + numpy.random.normal(size=(5, 5), scale=5)

    interp_mapx = cv2.resize(mapx, (80, 80))[8:72, 8:72].astype('float32')
    interp_mapy = cv2.resize(mapy, (80, 80))[8:72, 8:72].astype('float32')

    warped_image = cv2.remap(image, interp_mapx, interp_mapy, cv2.INTER_LINEAR)

    src_points = numpy.stack([mapx.ravel(), mapy.ravel()], axis=-1)
    dst_points = numpy.mgrid[0:65:16, 0:65:16].T.reshape(-1, 2)
    mat = umeyama(src_points, dst_points, True)[0:2]

    target_image = cv2.warpAffine(image, mat, (64, 64))

    return warped_image, target_image

images_A = get_image_paths("data/trump")
images_B = get_image_paths("data/cage")
images_A = load_images(images_A) / 255.0
images_B = load_images(images_B) / 255.0

# print(images_A.shape)

for i, image in enumerate(images_A) :
    # print(image.shape)
    _, target_image = random_warp(image)
    # print(target_image.shape)
    print("resized_data/trump/" + str(i) + ".png")
    cv2.imwrite("resized_data/trump/" + str(i) + ".png", target_image  * 255)
    
for i, image in enumerate(images_B) :
    _, target_image = random_warp(image)
    cv2.imwrite("resized_data/cage/" + str(i) + ".png", target_image  * 255)