
from cv2 import imshow
import numpy as np
import pandas as pd
import json

import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from matplotlib.patches import Patch
import webcolors

# EDA
def EDA_S(anns_file_path):
    # Read annotations
    with open(anns_file_path, 'r') as f:
        dataset = json.loads(f.read())

    categories = dataset['categories']
    anns = dataset['annotations']
    imgs = dataset['images']
    nr_cats = len(categories)
    nr_annotations = len(anns)
    nr_images = len(imgs)

    # Load categories and super categories
    cat_names = []
    super_cat_names = []
    super_cat_ids = {}
    super_cat_last_name = ''
    nr_super_cats = 0
    for cat_it in categories:
        cat_names.append(cat_it['name'])
        super_cat_name = cat_it['supercategory']
        # Adding new supercat
        if super_cat_name != super_cat_last_name:
            super_cat_names.append(super_cat_name)
            super_cat_ids[super_cat_name] = nr_super_cats
            super_cat_last_name = super_cat_name
            nr_super_cats += 1

    print('Number of super categories:', nr_super_cats)
    print('Number of categories:', nr_cats)
    print('Number of annotations:', nr_annotations)
    print('Number of images:', nr_images)

    # Count annotations
    cat_histogram = np.zeros(nr_cats,dtype=int)
    for ann in anns:
        cat_histogram[ann['category_id']-1] += 1

    # Initialize the matplotlib figure
    #f, ax = plt.subplots(figsize=(5,5))

    # Convert to DataFrame
    df = pd.DataFrame({'Categories': cat_names, 'Number of annotations': cat_histogram})
    df = df.sort_values('Number of annotations', 0, False)

    # Plot the histogram
    #plt.title("category distribution of train_all set ")
    #plot_1 = sns.barplot(x="Number of annotations", y="Categories", data=df, label="Total", color="b")

    # category labeling 
    sorted_temp_df = df.sort_index()

    # background = 0 에 해당되는 label 추가 후 기존들을 모두 label + 1 로 설정
    sorted_df = pd.DataFrame(["Backgroud"], columns = ["Categories"])
    sorted_df = sorted_df.append(sorted_temp_df, ignore_index=True)

    return sorted_df


# color map 가져오기
class_colormap = pd.read_csv("class_dict.csv")

def create_trash_label_colormap():
    """Creates a label colormap used in Trash segmentation.
    Returns:
        A colormap for visualizing segmentation results.
    """
    colormap = np.zeros((11, 3), dtype=np.uint8)
    for inex, (_, r, g, b) in enumerate(class_colormap.values):
        colormap[inex] = [r, g, b]

    return colormap

def label_to_color_image(label):
    """Adds color defined by the dataset colormap to the label.

    Args:
        label: A 2D array with integer type, storing the segmentation label.

    Returns:
        result: A 2D array with floating type. The element of the array
                is the color indexed by the corresponding element in the input label
                to the trash color map.

    Raises:
        ValueError: If label is not of rank 2 or its value is larger than color
              map maximum entry.
    """
    if label.ndim != 2:
        raise ValueError('Expect 2-D input label')

    colormap = create_trash_label_colormap()

    if np.max(label) >= len(colormap):
        raise ValueError('label value too large.')

    return colormap[label]


def loader_output(train_loader,val_loader,test_loader):
# train_loader의 output 결과(image 및 mask) 확인
    for imgs, masks, image_infos in train_loader:
        image_infos = image_infos[0]
        temp_images = imgs
        temp_masks = masks
        break

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 12))

    print('image shape:', list(temp_images[0].shape))
    print('mask shape: ', list(temp_masks[0].shape))
    print('Unique values, category of transformed mask : \n', [{int(i),category_names[int(i)]} for i in list(np.unique(temp_masks[0]))])

    ax1.imshow(temp_images[0].permute([1,2,0]))
    ax1.grid(False)
    ax1.set_title("input image : {}".format(image_infos['file_name']), fontsize = 15)

    ax2.imshow(temp_masks[0])
    ax2.grid(False)
    ax2.set_title("masks : {}".format(image_infos['file_name']), fontsize = 15)

    plt.show()


# val_loader의 output 결과(image 및 mask) 확인
    for imgs, masks, image_infos in val_loader:
        image_infos = image_infos[0]
        temp_images = imgs
        temp_masks = masks
        
        break
    
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 12))
    
    print('image shape:', list(temp_images[0].shape))
    print('mask shape: ', list(temp_masks[0].shape))
    
    print('Unique values, category of transformed mask : \n', [{int(i),category_names[int(i)]} for i in list(np.unique(temp_masks[0]))])
    
    ax1.imshow(temp_images[0].permute([1,2,0]))
    ax1.grid(False)
    ax1.set_title("input image : {}".format(image_infos['file_name']), fontsize = 15)
    
    ax2.imshow(temp_masks[0])
    ax2.grid(False)
    ax2.set_title("masks : {}".format(image_infos['file_name']), fontsize = 15)
    
    plt.show()

# test_loader의 output 결과(image) 확인
    for imgs, image_infos in test_loader:
        image_infos = image_infos[0]
        temp_images = imgs

        break

    fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(6, 6))

    print('image shape:', list(temp_images[0].shape))

    ax1.imshow(temp_images[0].permute([1,2,0]))
    ax1.grid(False)
    ax1.set_title("input image : {}".format(image_infos['file_name']), fontsize = 15)

    plt.show()

