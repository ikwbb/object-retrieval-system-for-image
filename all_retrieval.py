import numpy as np
import os
import cv2
from sklearn.metrics.pairwise import cosine_similarity
from matplotlib import pyplot as plt

IMG_COUNT = 10



def similarity(query_feat, gallery_feat):
    sim = cosine_similarity(query_feat.reshape(1, -1), gallery_feat.reshape(1, -1))
    sim = np.squeeze(sim)
    return sim

def retrival_idx(query_path, gallery_dir):
    query_feat = np.load(query_path)
    dict = {}
    for gallery_file in os.listdir(gallery_dir):
        gallery_feat = np.load(os.path.join(gallery_dir, gallery_file))
        gallery_idx = gallery_file.split('_')[0] + '.jpg'
        sim = similarity(query_feat, gallery_feat)
        dict[gallery_idx] = max(dict.get(gallery_idx, sim), sim)
    sorted_dict = sorted(dict.items(), key=lambda item: item[1]) 
    best_five = sorted_dict[-IMG_COUNT:] 
    return best_five

def visulization(retrived, query):
    plt.figure(figsize=(20, 30))
    plt.subplot(4, 4, 1)
    plt.title('query')
    query_img = cv2.imread(query)
    img_rgb_rgb = query_img[:,:,::-1]
    plt.imshow(img_rgb_rgb)
    for i in range(IMG_COUNT):
        img_path = './data/gallery/' + retrived[i][0]
        img = cv2.imread(img_path)
        img_rgb = img[:,:,::-1]
        plt.subplot(4, 4, i+2)
        plt.title(retrived[i][1])
        plt.imshow(img_rgb)
    plt.show()


def print_result(retrived, query_img_filename):
    return f"Q{query_img_filename.split('.')[0]}: {' '.join([img[0].split('.')[0] for img in retrived])}"



def process_query_images():
    result = []
    query_feat_dir = './data/all/query_feat'
    query_images_dir = './data/all/query'
    gallery_dir = './data/gallery_feature/'

    
    for feat_filename in os.listdir(query_feat_dir):
        if feat_filename.endswith('.npy'):
            
            query_feat_path = os.path.join(query_feat_dir, feat_filename)

            
            best_five = retrival_idx(query_feat_path, gallery_dir)
            print(best_five)
            best_five.reverse()

            
            
            query_img_filename = feat_filename.replace('_feats.npy', '.jpg')
            query_img_path = os.path.join(query_images_dir, query_img_filename)

            result.append(print_result(best_five, query_img_filename))

            
            
    
    with open('answer.txt', 'w') as file:
        file.write("\n".join(result))

if __name__ == '__main__':
    process_query_images()