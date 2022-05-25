import json
import os
import random
from PIL import Image
from tqdm import tqdm
import argparse


def sampling(dp: str, fp: str, bp: str, limit: int, save_path: str):
    with open(dp, 'r', encoding='utf-8') as f:
        lines: list = json.loads(f.readline())
    fonts = os.listdir(fp)
    bgs = []
    images = []
    for bg in os.listdir(bp):
        bgs.append(bg)
        images.append(Image.open(os.path.join(bp, bg)).convert("RGB"))
    results = []
    for _ in tqdm(range(limit)):
        bg_index = random.randint(0, len(bgs) - 1)
        width, height = images[bg_index].size
        start = [random.randint(0, width // 4 * 3), random.randint(0, height // 4)]
        ls = [lines[random.randint(0, len(lines) - 1)].strip("\n").strip("\r\t") for _ in range(random.randint(3, 5))]
        line_space = [random.randint(0, 20), random.randint(0, 20)]
        word_space = []
        font_paths = []
        font_sizes = []
        colors = []
        angles = []
        for l in ls:
            font_paths.append(fonts[random.randint(0, len(fonts) - 1)])
            fs = random.randint(10, 50) * 2
            font_sizes.append(fs)
            word_space.append([fs * 0.5, fs * random.randint(0, 10) // 10])
            tmp = l.split(" ")
            color = []
            angle = []
            for _ in range(len(tmp)):
                color.append((
                    random.randint(0, 255),
                    random.randint(0, 255),
                    random.randint(0, 255),
                ))
                angle.append(random.randint(-45, 45))
            colors.append(color)
            angles.append(angle)
        results.append({
            "bg_path": bgs[bg_index],
            "start": start,
            "lines": ls,
            "word_space": word_space,
            "line_space": line_space,
            "colors": colors,
            "angles": angles,
            "font_paths": font_paths,
            "font_sizes": font_sizes
        })

    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(json.dumps(results))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--background", type=str, help="background path")
    parser.add_argument("-f", "--font_path", type=str, help="font path")
    parser.add_argument("-d", "--dict_path", type=str, help="dict path")
    parser.add_argument("-l", "--limit", type=int, help="total sampling")
    parser.add_argument("-s", "--save_path", type=str, help="save path")
    arg = parser.parse_args()
    sampling(arg.dict_path, arg.font_path, arg.background, arg.limit, arg.save_path)
