import cv2, os

images_dir = "/home/inti/Desktop/Claudio/Semantic-Segmentation-Suite/crack_images_v2/"

images_subdir = images_dir + "train_labels/"
images_list = [file for file in os.listdir(images_subdir)
    if os.path.isfile(images_subdir + file)]
output_dir = images_subdir + "fixed/"
if not os.path.isdir(output_dir):
    os.makedirs(output_dir)
try:
    for image in images_list:
        # count = 0
        # print("Checking image {}.".format(image))
        image_cv = cv2.imread(images_subdir + image,
                           cv2.IMREAD_GRAYSCALE)
        for i in range(image_cv.shape[0]):
            for j in range(image_cv.shape[1]):
                if image_cv[i][j] >= 250:
                    image_cv[i][j] = 255
                else:
                    image_cv[i][j] = 0
                """
                if image[i][j] not in [0, 255]:
                    # count += 1
                    # print("Found a {!s} value".format(image[i][j]))
                """
        cv2.imwrite(output_dir + image.split('.')[0] + ".png", image_cv)

        # print("Found {!s} non 0/255 values".format(count), end='')
        # st = input()
        # if st != "":
        #     raise KeyboardInterrupt
except KeyboardInterrupt:
    print("Operation cancelled.")

images_subdir = images_dir + "val_labels/"
images_list = [file for file in os.listdir(images_subdir)
    if os.path.isfile(images_subdir + file)]
output_dir = images_subdir + "fixed/"
if not os.path.isdir(output_dir):
    os.makedirs(output_dir)
try:
    for image in images_list:
        # count = 0
        # print("Checking image {}.".format(image))
        image_cv = cv2.imread(images_subdir + image,
                           cv2.IMREAD_GRAYSCALE)
        for i in range(image_cv.shape[0]):
            for j in range(image_cv.shape[1]):
                if image_cv[i][j] >= 250:
                    image_cv[i][j] = 255
                else:
                    image_cv[i][j] = 0
                """
                if image[i][j] not in [0, 255]:
                    # count += 1
                    # print("Found a {!s} value".format(image[i][j]))
                """
        cv2.imwrite(output_dir + image.split('.')[0] + ".png", image_cv)

        # print("Found {!s} non 0/255 values".format(count), end='')
        # st = input()
        # if st != "":
        #     raise KeyboardInterrupt
except KeyboardInterrupt:
    print("Operation cancelled.")

images_subdir = images_dir + "test_labels/"
images_list = [file for file in os.listdir(images_subdir)
    if os.path.isfile(images_subdir + file)]
output_dir = images_subdir + "fixed/"
if not os.path.isdir(output_dir):
    os.makedirs(output_dir)
try:
    for image in images_list:
        # count = 0
        # print("Checking image {}.".format(image))
        image_cv = cv2.imread(images_subdir + image,
                           cv2.IMREAD_GRAYSCALE)
        for i in range(image_cv.shape[0]):
            for j in range(image_cv.shape[1]):
                if image_cv[i][j] >= 250:
                    image_cv[i][j] = 255
                else:
                    image_cv[i][j] = 0
                """
                if image[i][j] not in [0, 255]:
                    # count += 1
                    # print("Found a {!s} value".format(image[i][j]))
                """
        cv2.imwrite(output_dir + image.split('.')[0] + ".png", image_cv)

        # print("Found {!s} non 0/255 values".format(count), end='')
        # st = input()
        # if st != "":
        #     raise KeyboardInterrupt
except KeyboardInterrupt:
    print("Operation cancelled.")