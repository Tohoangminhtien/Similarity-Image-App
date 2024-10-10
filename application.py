from flask import Flask, render_template, request, url_for
import base64
from tqdm import tqdm
import requests
from utils import get_array, connect_vector_db, insert_vector_db, delete_all_db
from transformers import TFViTModel, AutoImageProcessor
import numpy as np
from io import BytesIO
import cv2
import tensorflow as tf
import pandas as pd
import random

app = Flask(__name__)

pretrained = 'google/vit-base-patch16-224-in21k'
image_processor = AutoImageProcessor.from_pretrained(pretrained)
vit_model = TFViTModel.from_pretrained(pretrained)


@app.route('/', methods=['GET'])
def default_get():
    return render_template('index.html')


@app.route('/', methods=['POST'])
def default_post():
    # ---------------ĐỌC DỮ LIỆU POST----------------
    url_query = request.form['url-query']
    image_query = request.files['image']
    numquery = int(request.form['hidden-numquery'])
    imgbase64 = request.form['imgbase64']

    # ---------------XỬ LÝ ẢNH-----------------------
    if not url_query and not image_query and not imgbase64:
        return render_template('index.html', notification='No url or file selected')
    if imgbase64:
        encoded_data = imgbase64.split(',')[1]
        decoded_data = base64.b64decode(encoded_data)
        nparr = np.frombuffer(decoded_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    else:
        if url_query:
            response = requests.get(url_query)
            image_array = np.asarray(
                bytearray(response.content), dtype="uint8")
            img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        if image_query:
            dataBytes = image_query.read()
            arrayFlatten = np.frombuffer(dataBytes, dtype=np.uint8)
            img = cv2.imdecode(arrayFlatten, cv2.IMREAD_COLOR)

    # --------------SAVE LẠI HISTORY---------------
    cv2.imwrite('static/img/last_image_query.jpg', img)

    # --------------TÍNH VECTOR--------------------
    query_vec = vit_model(image_processor(img, return_tensors="np")[
                          'pixel_values'])['pooler_output'].numpy()[0].tolist()

    # --------------SEARCH--------------------
    client = connect_vector_db('secret_key.json')
    res = client.search(
        collection_name="ViT",
        data=[query_vec],
        limit=numquery,
        search_params={"metric_type": "COSINE"},
        output_fields=['url', 'describe']
    )
    link_result = [res[0][i]['entity']['url'] for i in range(numquery)]
    describe_list = [res[0][i]['entity']['describe'] for i in range(numquery)]
    distances = [round(res[0][i]['distance'], 3) for i in range(numquery)]
    prices = [round(random.uniform(1, 200), 2) for _ in range(numquery)]
    products_data = zip(link_result, describe_list, distances, prices)
    client.close()

    return render_template('index.html', products_data=products_data, is_show=True)


@app.route('/delete', methods=['POST'])
def delete_all():
    client = connect_vector_db('secret_key.json')
    result = delete_all_db(client, 'ViT')
    client.close()
    delete_count = result['delete_count']
    return render_template('upload.html', notification=f'Remove {delete_count} rows')


@app.route('/upload', methods=['GET'])
def upload_page_get():
    return render_template('upload.html')


@app.route('/upload', methods=['POST'])
def upload_page_post():
    # --------------Check IO error-------------
    if 'file' not in request.files:
        return render_template('upload.html', notification=f'No file part')
    file = request.files['file']
    if file.filename == '':
        return render_template('upload.html', notification=f'No selected file')

    if file and file.filename.endswith('.xlsx'):
        # -----------ĐỌC FILE EXCEL-------------
        df = pd.read_excel(BytesIO(file.read()))
        url_list = df['URL'].to_list()
        describe_list = df['Describe'].to_list()

        # -----------LẤY ẢNH TỪ URL v1.2--------------
        tensor = []
        for url in tqdm(url_list, desc="Processing Images"):
            img = get_array(url)
            process = image_processor(img, return_tensors="np")['pixel_values']
            tensor.append(process)

        tensor = np.array(tensor)
        print(tensor.shape)
        print('Get image: DONE')

        # ------------CALCULATE VECTOR v1.2------------
        n_samples = tensor.shape[0]
        predictions = []
        batch_size = 32

        for start in tqdm(range(0, n_samples, batch_size), desc="Calculating Vectors"):
            end = min(start + batch_size, n_samples)
            batch_predictions = vit_model(tensor[start:end])[
                'pooler_output'].numpy()
            # (32, 768)
            predictions.append(batch_predictions)

        predictions = np.vstack(predictions)
        # (n, 768)
        print("VECTOR CALCULATED")

        # --------------INSERT TO DB-----------------
        insert_data = []
        for vector, url, describe in zip(predictions, url_list, describe_list):
            insert_data.append({
                'vector': vector.tolist(),
                'url': url,
                'describe': describe
            })

        client = connect_vector_db('secret_key.json')
        result = insert_vector_db(client, 'MobileNet', insert_data)
        client.close()
        print("INSERTED TO DB")

    return render_template('upload.html', notification=f'Insert {len(insert_data)} rows into database')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
