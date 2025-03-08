# Object Retrieval System in Image

This repository contains an implementation of an image retrieval system designed to search for and rank similar object in images from a large gallery based on query images. The system leverages CNN-based feature extraction to enhance retrieval performance. It is built to process a gallery of images and return ranked results based on similarity to a given query.

## Example
![Retrieving an object from other images](images/object_retrieval_example.png)


## Features
  1. A CNN-based approach using a pre-trained ResNet50 model for feature extraction, paired with cosine similarity for ranking.
- **Modular Design**: Code is organized into separate folders for each method, making it easy to extend or modify.

## Repository Structure
  - `extract_feature_multi_process.py`: Feature extraction using ResNet50.
  - `all_retrieval.py`: Image retrieval and ranking using cosine similarity.


## Methodology

### Method 1: CNN-Based Retrieval (ResNet50)
This method extracts deep features from images using a pre-trained ResNet50 model and ranks gallery images based on cosine similarity.

- **Feature Extraction**:
  - Images are pre-processed (BGR to RGB conversion) using OpenCV.
  - Objects are cropped using a Fast R-CNN model when multiple instances are present.
  - ResNet50 extracts features after removing the fully connected layer, followed by global average pooling to generate a 1D feature vector.
- **Retrieval**:
  - Cosine similarity is computed between the query feature vector and gallery feature vectors.
  - Top 10 images with the highest similarity scores are returned.


## Setup and Usage

### Prerequisites
- Python 3.x
- Libraries: OpenCV, NumPy, PyTorch (for ResNet50), and other dependencies (see `requirements.txt` if provided).
- A gallery of images (not included in this repo).

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/image-retrieval-system.git
   cd image-retrieval-system
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Add your gallery images to the `gallery/` folder and query images to the `query/` folder.

### Running the Code
- For CNN-based retrieval:
  ```bash
  python /extract_feature_multi_process.py
  python /all_retrieval.py
  ```
  

## Notes
- The `gallery/` folder is empty to save space. Populate it with your image dataset before running the code.
- Query images and bounding box annotations (if required) should be placed in the `query/` folder.
- Adjust file paths in the scripts if your dataset structure differs.
