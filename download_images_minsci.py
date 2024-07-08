import pandas as pd
import numpy as np
import requests
from PIL import Image
import io
import time
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
import json
import argparse

def requests_PIL_download(record):
    asset = record['asset_name']
    arkid = record['image_arkid'].split('/')[-1]
    MAX_SIZE = 250
    image_url = f'https://ids.si.edu/ids/deliveryService?id={asset}&max={MAX_SIZE}'
    width, height = np.nan, np.nan
    filename = 'minsci/{}.jpg'.format(arkid)

    try:
        r = requests.get(image_url, timeout=60)
        if r.headers['Content-Type'] == 'image/jpeg':
            try:
                with Image.open(io.BytesIO(r.content)) as im:
                    width, height = im.size
                    im.save(filename)
            except:
                print('Weird error with ' + asset)
    except:
        print('Timeout error with ' + asset)
    return {'width': width, 'height': height, 'arkid': arkid}


ap = argparse.ArgumentParser()
ap.add_argument('-t', "--image_tsv", required=True,
                help="file path containing image data in TSV format")
ap.add_argument("-p", "--processes",
                help="number of processes")
ap.add_argument("-d", "--dim-file",
                help="file path for dimension tsv output")
args = ap.parse_args()

image_df = pd.read_csv(args.image_tsv, sep='\t')
image_ids = image_df.to_dict(orient='records')

start_time = time.perf_counter()

dimension_list = []

with ThreadPoolExecutor(max_workers=int(args.processes)) as executor:
    dimension_list = list(executor.map(requests_PIL_download, image_ids))

end_time = time.perf_counter()
elapsed_time = end_time - start_time
print(f"Downloaded {len(dimension_list)} images in {elapsed_time} s")

dimension_df = pd.DataFrame(dimension_list)
dimension_df.to_csv(args.dim_file, index=False, sep='\t')