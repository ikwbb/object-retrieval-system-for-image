import torch
import cv2
import os
import numpy as np
import torchvision.models as models
import torchvision.transforms as transforms
from tqdm import tqdm
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import sys


def query_crop(query_path, txt_path, save_path):
    query_img = cv2.imread(query_path)
    query_img = query_img[:,:,::-1] 
    txt = np.loadtxt(txt_path)     
    crop = query_img[int(txt[1]):int(txt[1] + txt[3]), int(txt[0]):int(txt[0] + txt[2]), :] 
    cv2.imwrite(save_path, crop[:,:,::-1])  
    return crop

def resnet_extraction(img, featsave_path):
    
    resnet_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])
    img_transform = resnet_transform(img)
    img_transform = torch.unsqueeze(img_transform, 0)

    
    resnet = models.resnet50(pretrained=True)
    resnet.eval()  

    
    modules = list(resnet.children())[:-2]
    resnet_feat_extractor = torch.nn.Sequential(*modules)

    with torch.no_grad():  
        feats = resnet_feat_extractor(img_transform)  

    
    pooled_feats = F.adaptive_avg_pool2d(feats, (1, 1))
    feats_np = pooled_feats.cpu().squeeze().numpy()
    np.save(featsave_path, feats_np)


def get_object_detection_model(pretrained=True):
    
    model = fasterrcnn_resnet50_fpn(pretrained=pretrained)
    model.eval()
    return model

def detect_and_crop_objects(img, model, detection_threshold=0.5):
    transform = T.Compose([T.ToTensor()])
    img_tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        predictions = model(img_tensor)

    
    pred_boxes = predictions[0]['boxes']
    pred_scores = predictions[0]['scores']
    pred_labels = predictions[0]['labels']

    
    high_scores_idx = torch.where(pred_scores > detection_threshold)[0]
    pred_boxes = pred_boxes[high_scores_idx]

    cropped_images = []
    for box in pred_boxes:
        
        box = box.to(torch.int64)
        cropped_img = img[box[1]:box[3], box[0]:box[2], :]
        cropped_images.append(cropped_img)

    return cropped_images

def feat_extractor_gallery(gallery_dir, feat_savedir, detection_model, process_id=None, num_processes=1):
    image_files = os.listdir(gallery_dir)
    
    portion = len(image_files) // num_processes
    start = process_id * portion
    end = (process_id + 1) * portion if process_id != num_processes - 1 else len(image_files)
    portion_files = image_files[start:end]
    
    for img_file in tqdm(portion_files, desc=f"Process {process_id}"):
        img_path = os.path.join(gallery_dir, img_file)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  

        
        cropped_objects = detect_and_crop_objects(img, detection_model)

        
        for idx, crop in enumerate(cropped_objects):
            img_resize = cv2.resize(crop, (224, 224), interpolation=cv2.INTER_CUBIC)  
            featsave_path = os.path.join(feat_savedir, f"{img_file.split('.')[0]}_obj{idx}.npy")
            resnet_extraction(img_resize, featsave_path)



def feat_extractor_query():
    query_path = './data/query/query.jpg'
    txt_path = './data/query_txt/query.txt'
    save_path = './data/cropped_query/query.jpg'
    featsave_path = './data/query_feat/query_feats.npy'
    crop = query_crop(query_path, txt_path, save_path)
    crop_resize = cv2.resize(crop, (224, 224), interpolation=cv2.INTER_CUBIC)
    resnet_extraction(crop_resize, featsave_path)

def main(process_id, num_processes):
    feat_extractor_query()
    gallery_dir = './data/gallery/'
    feat_savedir = './data/gallery_feature/'
    detection_model = get_object_detection_model(pretrained=True)
    feat_extractor_gallery(gallery_dir, feat_savedir, detection_model, process_id, num_processes)

if __name__=='__main__':
    if len(sys.argv) != 3:
        print("Usage: python extract_feature_multi_process.py [process_id] [num_processes]")
        sys.exit(1)
    
    process_id = int(sys.argv[1])
    num_processes = int(sys.argv[2])

    main(process_id, num_processes)