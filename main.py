# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "pillow",
#     "requests",
#     "numpy",
#     "matplotlib",
#     "dotenv"
# ]
# ///

import requests as req
import PIL as Image
import numpy as np 
import matplotlib as mpl
from dotenv import dotenv_values
from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
import torch.nn as nn


import os
from huggingface_hub import InferenceClient

print('test')

config = dotenv_values(".env")  # config = {"USER": "foo", "EMAIL": "foo@example.org"}
print(config['SECRET_KEY'])


HF_TOKEN = config['SECRET_KEY']
#model = "gpt2"  # or "facebook/bart-large-cnn", etc.
#headers = {"Authorization": f"Bearer {HF_TOKEN}"}
#data = {"inputs": "Write a short haiku about rain."}
#resp = req.post(f"https://router.huggingface.co/hf-inference/{model}", headers=headers, json=data)
#resp.raise_for_status()
#print(resp.json())

client = InferenceClient(api_key=HF_TOKEN)

completion = client.chat.completions.create(
    model="openai/gpt-oss-120b",
    messages=[
        {
            "role": "user",
            "content": "How many 'g's in 'huggingface'?"
        }
    ],
)

print(completion.choices[0].message)


# processor = SegformerImageProcessor.from_pretrained("sayeed99/segformer_b3_clothes")
# model = AutoModelForSemanticSegmentation.from_pretrained("sayeed99/segformer_b3_clothes")

# url = "https://plus.unsplash.com/premium_photo-1673210886161-bfcc40f54d1f?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxzZWFyY2h8MXx8cGVyc29uJTIwc3RhbmRpbmd8ZW58MHx8MHx8&w=1000&q=80"

# image = Image.open(req.get(url, stream=True).raw)
# inputs = processor(images=image, return_tensors="pt")

# outputs = model(**inputs)
# logits = outputs.logits.cpu()

# upsampled_logits = nn.functional.interpolate(
#     logits,
#     size=image.size[::-1],
#     mode="bilinear",
#     align_corners=False,
# )

# pred_seg = upsampled_logits.argmax(dim=1)[0]
# mpl.imshow(pred_seg)
