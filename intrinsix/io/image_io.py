import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import warnings

import cv2
from PIL import Image
import numpy as np
from torchvision import transforms


def load_image(path):
    try:
        extension = os.path.splitext(path)[1].lower()
        if extension in ['.png', '.jpg', '.jpeg']:
            return load_ldr_image(path)
        elif extension in ['.exr']:
            return load_exr_image(path)
        elif extension in ['.npy']:
            return np.load(path)
        elif extension in ['.npz']:
            return np.load(path)['features']
    except Exception:
        warnings.warn(f"Unable to load {path}")
        raise


def save_image(img, path):
    extension = os.path.splitext(path)[1].lower()
    if extension == '.exr':
        save_exr_image(img, path)
    elif extension in ['.npy']:
        return save_np_image(img, path, compressed=False)
    elif extension in ['.npz']:
        return save_np_image(img, path, compressed=True)
    else:
        save_ldr_image(img, path)


def load_ldr_image(path):
    image = cv2.imread(path)

    # Do not forget that OpenCV read images in BGR order.
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Normalize source images to [0, 1].
    image = image.astype(np.float32) / 255.0

    return image


def save_ldr_image(img, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    image = Image.fromarray((img * 255).astype(np.uint8).squeeze())
    image.save(path)


def load_exr_image(path):
    image = cv2.imread(path,  cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)

    # Do not forget that OpenCV read images in BGR order.
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Normalize source images to [0, 1].
    image = image.astype(np.float32)

    return image


def save_exr_image(img, path):
    import imageio as imageio
    os.makedirs(os.path.dirname(path), exist_ok=True)
    imageio.imwrite(path, img.astype(np.float32).squeeze())
    # imageio.imwrite(path, img.astype(np.float32).squeeze(), flags=0x001)


def save_np_image(img, path, compressed=False):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if compressed:
        np.savez_compressed(path, features=img)
    else:
        np.save(path, img)


def show_image(image, normalize=False):
    if normalize:
        image = image / image.max()

    to_pil = transforms.ToPILImage()
    if isinstance(image, np.ndarray):
        if image.dtype in (np.float32, np.float64):
            image = (image*255).astype(np.uint8)
    im = to_pil(image)
    im.show()


def np_to_cv(data):
    return (data * 255).astype(np.uint8)[...,::-1].copy()


def np_to_pil(data):
    return Image.fromarray((data * 255).astype(np.uint8).copy())
