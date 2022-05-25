import os.path
import numpy as np
from PIL import Image, ImageFont, ImageDraw


def draw_word(bg: Image,
              words: list,
              font: ImageFont,
              start: np.ndarray,
              space: np.ndarray,
              colors: list,
              angle: list):
    bbox = []
    last = [bg.size[0], 0]
    for i, word in enumerate(words):
        x1, y1 = start if len(bbox) == 0 else np.array(bbox[len(bbox) - 1]['polygon'][3]) + space
        x1, y1 = int(x1), int(y1)
        word_size = font.getsize(text=word)
        new_image = Image.new("RGBA", word_size, (0, 0, 0, 0))
        drawer = ImageDraw.Draw(new_image)
        drawer.text((0, 0), word, tuple(colors[i]), font)
        new_image = new_image.rotate(angle[i], expand=True)
        w, h = new_image.size
        x2, y2 = x1 + w, y1 + h
        if (np.array([x2, y2]) < bg.size).all():
            last[0] = min([last[0], x1])
            last[1] = max([last[1], y2])
            bbox.append({
                "polygon": [[x1, y1], [x1, y2],
                            [x2, y2], [x2, y1]],
                "label": word
            })
            bg.paste(new_image, (x1, y1, x2, y2), new_image)
    if len(bbox) == 0:
        raise Exception("Image ({}, {}) can't contain word having {} size!".format(*bg.size, font.size))
    return bg, bbox, np.array(last)


def generator(bg_root: str,
              font_root: str,
              bg_path: str,
              start: list,
              lines: list,
              word_space: list,
              line_space: list,
              colors: list,
              angles: list,
              font_paths: list,
              font_sizes: list):
    bg = Image.open(os.path.join(bg_root, bg_path)).convert("RGB")
    start = np.array(start)
    word_space = np.array(word_space)
    line_space = np.array(line_space)
    word_boxes = []
    for i, line in enumerate(lines):
        try:
            font = ImageFont.truetype(os.path.join(font_root, font_paths[i]), font_sizes[i], encoding='utf-8')
            bg, word_box, last = draw_word(bg,
                                           line.split(" "),
                                           font,
                                           start,
                                           word_space[i],
                                           colors[i],
                                           angles[i])
            start = last + line_space
            word_boxes.extend(word_box)
        except Exception as e:
            pass
    return np.asarray(bg, dtype=np.uint8), word_boxes
