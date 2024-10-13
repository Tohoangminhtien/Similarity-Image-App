# Image Similarity Search using Vision Transformer (ViT)
This project implements an image similarity search system using a Vision Transformer (ViT) model. The system retrieves images similar to a query image from a dataset by encoding the images into feature vectors and comparing them.

## Feature
- Vision Transformer (ViT): Utilizes a pre-trained ViT model for image encoding.
- Image Similarity Search: Computes similarity between images using feature vectors.
- Efficient Retrieval: Uses nearest neighbors search Zilliz Cloud to retrieve similar images quickly.

## Installation
### Prerequisites
Ensure you have the following installed on your machine:
- Python >= 3.8
- pip package manager
- CUDA (optional, for GPU acceleration)

### Step
1. Clone repository:
   ```bash
   git clone https://github.com/Tohoangminhtien/Similarity-Image-App.git
   cd Similarity-Image-App

2. Install the required packages:
   ```bash
      pip install -r requirements.txt
   ```
## Usage
### Running the Web Service
Start the web service for querying images via an API:
```bash
python application.py
```
The service will be available at http://127.0.0.1:5000/.

### Pull Docker Image from Docker Hub
You can pull the pre-built Docker image from Docker Hub and run it directly.

1. Pull the Docker image
   ```bash
   docker pull tienthm/similar-image-app:v1.0
   ```
2. Run the container
   ```bash
   docker run -p 5000:5000 tienthm/similar-image-app:v1.0
   ```
