import argparse
import sys

from src.system import face_recognition_system

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('model', type=str, help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file', default='models/83_epochs/')
    # parser.add_argument('--option', type=str, help='Could be `demo_via_img` or `demo_via_cam`', default='demo_via_img')
    # parser.add_argument('--dataset_path', type=str, help='Path to the dataset.', default='datasets/')
    parser.add_argument('--cmnd_file', type=str, help='ID card with face for demo.')
    parser.add_argument('--image_file', type=str, help='Images to demo with.')

    return parser.parse_args(argv)

def main():
    args = parse_arguments(sys.argv[1:])

    my_system = face_recognition_system(args.model)

    if args.image_file or args.cmnd_file:
        my_system.verify_face_via_image(args.cmnd_file, args.image_file)
    else:
        print("Please specify the image file.")

if __name__ == "__main__":
    main()