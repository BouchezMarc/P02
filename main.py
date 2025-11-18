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
import PIL as pillow
#import pil
import numpy as np 
import matplotlib as mpl
#import dotenv as dotenv
from dotenv import dotenv_values

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