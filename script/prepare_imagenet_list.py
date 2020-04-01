import os
from misc import dump_pickle
import argparse

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    return classes


def prepare_images_list(data_dir, dump_path):
    classes = find_classes(data_dir)
    data_images_list = []
    for i, class_name in enumerate(classes):
        print('processing %d-th class: %s' % (i, class_name))
        temp = []
        class_dir = os.path.join(data_dir, class_name)
        filenames = os.listdir(class_dir)
        for filename in filenames:
            if is_image_file(filename):
                temp.append(os.path.join(class_dir, filename))

        data_images_list.append(temp)

    dump_pickle(data_images_list, dump_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--dump_path', type=str)
    args = parser.parse_args()
    prepare_images_list(args.data_dir, args.dump_path)