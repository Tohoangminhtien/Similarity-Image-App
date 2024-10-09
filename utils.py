import cv2
import numpy as np
import requests
from pymilvus import MilvusClient
import json


def get_array(url: str) -> np.ndarray:
    '''
    No preprocess, reshape, resize,...
    '''
    response = requests.get(url)
    image_array = np.asarray(bytearray(response.content), dtype="uint8")
    # Đọc ảnh bằng OpenCV
    img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    return img


def connect_vector_db(key_dir):
    # Đọc tệp JSON
    with open(key_dir, 'r') as file:
        config = json.load(file)

    cluster_endpoint = config['CLUSTER_ENDPOINT']
    token = config['TOKEN']
    client = MilvusClient(uri=cluster_endpoint, token=token)
    return client


def get_describe_db(client, collections):
    return client.describe_collection(collections)


def insert_vector_db(client, collection, data):
    return client.insert(collection, data=data)


def delete_all_db(client, collection):
    return client.delete(collection_name=collection, filter='Auto_id >= 0')
