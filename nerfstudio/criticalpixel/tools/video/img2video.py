from pathlib import Path
import argparse
import imageio
import cv2
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, required=True)
    parser.add_argument('--image_fmt', default='*.jpg')
    parser.add_argument('--output', type=str, required=True)
    return parser.parse_args()

def main(args):
    image_dir = Path(args.image_dir)
    image_paths = sorted(list(image_dir.glob(args.image_fmt)))

    writer = imageio.get_writer(args.output, fps=10, quality=8, macro_block_size=1)
    
    # 循环写入帧
    for image_path in tqdm(image_paths):
        img = cv2.imread(str(image_path))

        h, w = img.shape[:2]
        if h % 2 != 0:
            h += 1
            img = cv2.resize(img, (w, h))

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        writer.append_data(img)

if __name__ == '__main__':
    main(parse_args())
