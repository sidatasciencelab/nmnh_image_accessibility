from PIL import Image
import requests
import torch
from transformers import CLIPProcessor, CLIPModel
from pathlib import Path
import argparse
import time
import numpy as np
import pandas as pd

def embed_and_classify_file_batch(file_list, output_file):
    start_time = time.perf_counter() 
    batch_size = 32
    class_labels = ['A cross section of a rock', 
            'A photo of a rock']
    embed_list = []
    class_probs = []
    for start in range(0, len(file_list), batch_size):
        file_batch = file_list[start:start+batch_size]
        images = [Image.open(page_thumb) for page_thumb in file_batch]
        inputs = processor(text=class_labels, images=images, 
                        return_tensors="pt", padding=True).to(device)
        outputs = model(**inputs)
        image_embeds = outputs.image_embeds.squeeze(0).cpu().detach().numpy()
        embed_list.append(image_embeds)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1).cpu().detach().numpy()
        class_probs.append(probs)
    
    embeddings = np.vstack(embed_list)
    with open(f'minsci_embeddings/{output_file}.npy','wb') as np_file:
        np.save(np_file, embeddings)

    prob_stack = np.vstack(class_probs)
    prob_df = pd.DataFrame(prob_stack)
    prob_df.columns = ['cross-section','photo']
    prob_df['filename'] = [file.stem for file in file_list]
    prob_df.to_csv(f'minsci_probabilities/{output_file}.tsv', sep='\t', index=False)
    
    end_time = time.perf_counter() 
    elapsed_time = end_time - start_time    
    print(f'{len(embeddings)} embeddings created in {elapsed_time} s')

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--device",
                    default='cpu',
                    help="device used to encode")
    args = ap.parse_args()

    start_time = time.time()

    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    device = 'cpu'
    if args.device == 'gpu':
        if torch.cuda.is_available():
            device = 'cuda'
    
    model.to(device)

    end_time = time.time()
    model_load = end_time - start_time
    print(f'Model loaded in {model_load}')

    start_time = time.time()

    all_names = list(Path('minsci').rglob('*.jpg'))
    #all_names = all_names[:1015]
    print(len(all_names))

    end_time = time.time()
    image_load = end_time - start_time
    print(f'Images loaded in {image_load}')

    FILE_BATCH_SIZE = 1000
    for start in range(0, len(all_names), FILE_BATCH_SIZE):
        file_subset = all_names[start:start+FILE_BATCH_SIZE]
        if len(file_subset) < FILE_BATCH_SIZE:
            file_end = start + len(file_subset)
        else:
            file_end = start + FILE_BATCH_SIZE
        print(start, file_end)
        file_out = f'embeddings_{start}_{file_end}'
        embed_and_classify_file_batch(file_subset, file_out)
         