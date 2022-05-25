import argparse
import sys

try:
    from src.system import face_recognition_system
except:
    from system import face_recognition_system

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('model', type=str, help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file', default='/models/83_epochs/')
    parser.add_argument('--option', type=str, help='Could be `index` or `demo_via_img` or `demo_via_cam`', default='demo_via_img')
    parser.add_argument('--dataset_path', type=str, help='Path to the dataset.', default='datasets/')
    parser.add_argument('--image_file', type=str, nargs='+', help='Images to demo with.')

    return parser.parse_args(argv)

if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])

    my_system = face_recognition_system(args.model, args.dataset_path)

    if args.option == 'index':
        my_system.index_dataset()
    elif args.option == 'demo_via_img':
        if args.image_file:
            my_system.recognize_face_via_image(args.image_file)
        else:
            print("Please specify the image file.")
    elif args.option == 'demo_via_cam':
        my_system.recognize_face_via_cam()