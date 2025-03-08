import torch
import cv2
import os
import numpy as np
import torchvision.models as models
import torchvision.transforms as transforms
from tqdm import tqdm
import torch.nn.functional as F


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


def feat_extractor_query_all():
    
    query_dir = './data/all/query'
    txt_dir = './data/all/query_txt'
    cropped_query_dir = './data/all/cropped_query'
    featsave_dir = './data/all/query_feat'
    
    
    os.makedirs(cropped_query_dir, exist_ok=True)
    os.makedirs(featsave_dir, exist_ok=True)
    
    
    for filename in os.listdir(query_dir):
        if filename.endswith('.jpg'):  
            
            query_path = os.path.join(query_dir, filename)
            txt_path = os.path.join(txt_dir, filename.replace('.jpg', '.txt'))
            save_path = os.path.join(cropped_query_dir, filename)
            featsave_path = os.path.join(featsave_dir, filename.replace('.jpg', '_feats.npy'))
            
            
            crop = query_crop(query_path, txt_path, save_path)
            crop_resize = cv2.resize(crop, (224, 224), interpolation=cv2.INTER_CUBIC)
            resnet_extraction(crop_resize, featsave_path)

def main():
    feat_extractor_query_all()

if __name__=='__main__':
    main()