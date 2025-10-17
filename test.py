# test_hf_embeddings.py
import requests
import config  # make sure your HF_API_KEY is in this file

API_URL = f"https://api-inference.huggingface.co/models/BAAI/bge-base-en-v1.5"
headers = {"Authorization": f"Bearer {config.HF_API_KEY}"}

def get_embedding(text):
    response = requests.post(API_URL, headers=headers, json={"inputs": text})
    if response.status_code == 200:
        data = response.json()
        # Some models return nested lists, so flatten it if needed
        if isinstance(data, list) and isinstance(data[0], list):
            return data[0]
        return data
    else:
        print(f"❌ Error {response.status_code}: {response.text}")
        return None

text = "Vietnam is a beautiful country with rich culture and landscapes."
embedding = get_embedding(text)

if embedding:
    print(f"✅ Success! Got embedding of length: {len(embedding)}")
