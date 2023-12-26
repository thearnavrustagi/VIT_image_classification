from PIL import Image, ImageFilter
from tqdm import tqdm
import cv2
import numpy as np
import os

labels = (
    "combat destroyed_buildings fire human_aid_rehabilitation military_vehicles".split()
)


def imwrite(i, o, n, t):
    cv2.imwrite(os.path.join(o, f"{n}_{t}.png"), cv2.cvtColor(i, cv2.COLOR_BGR2RGB))
    return i


def process_and_save_images(name, image_path, output_directory):
    # Open the image
    original_image = cv2.resize(cv2.imread(image_path), (75, 75))

    # Create the output directory if it doesn't exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    vflip = cv2.flip(original_image, 0)
    imwrite(vflip, output_directory, name, "vflip")

    hflip = cv2.flip(original_image, 1)
    imwrite(hflip, output_directory, name, "hflip")
    tpose = imwrite(cv2.flip(original_image, -1), output_directory, name, "tpose")

    imgs = [original_image, vflip, hflip, tpose]

    def gnoise(img):
        gauss_noise = np.zeros(img.shape, dtype=np.uint8)
        cv2.randn(gauss_noise, (128, 128, 128), (20, 20, 20))
        gauss_noise = (gauss_noise * 0.5).astype(np.uint8)
        return cv2.add(img, gauss_noise)

    def adjust_gamma(image, gamma=1.0):
        invGamma = 1.0 / gamma
        table = np.array(
            [((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]
        ).astype("uint8")

        return cv2.LUT(image, table)

    imgs += [
        imwrite(gnoise(original_image), output_directory, name, "gn_og"),
        imwrite(gnoise(vflip), output_directory, name, "gn_vflip"),
        imwrite(gnoise(hflip), output_directory, name, "gn_hflip"),
        imwrite(gnoise(tpose), output_directory, name, "gn_tpose"),
    ]

    for i, img in enumerate(imgs[:]):
        brightened_image = adjust_gamma(img, 1.5)
        imgs.append(imwrite(brightened_image, output_directory, name, f"b_{i}"))
        dull = adjust_gamma(img, 0.5)
        imgs.append(imwrite(dull, output_directory, name, f"d_{i}"))


if __name__ == "__main__":
    for split in "train test".split():
        for label in labels:
            path = f"./raw_dset/{split}/{label}"
            out_path = f"./dset/{split}/{label}"
            for fname in tqdm(os.listdir(path)):
                image_path = f"{path}/{fname}"
                process_and_save_images(fname, image_path, out_path)
