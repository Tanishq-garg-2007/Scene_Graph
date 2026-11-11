import os, shutil, json

SELECTED_IMAGES = [
    '167.jpg', '29.jpg', '7.jpg', '193.jpg', '73.jpg', '65.jpg', '59.jpg', '36.jpg',
    '192.jpg', '27.jpg', '177.jpg', '142.jpg', '23.jpg', '154.jpg', '111.jpg', '9.jpg',
    '8.jpg', '24.jpg', '57.jpg', '62.jpg', '132.jpg', '158.jpg', '201.jpg', '146.jpg'
]

os.makedirs('dataset_upload', exist_ok=True)
for img in SELECTED_IMAGES:
    src = os.path.join('images', img)
    dst = os.path.join('dataset_upload', img)
    if os.path.exists(src):
        shutil.copy2(src, dst)

metadata = {
    "title": "Spatial Benchmark 24 Images",
    "id": "tanishqgarg2007/spatial-benchmark-24-images",
    "licenses": [{"name": "CC0-1.0"}]
}

with open('dataset_upload/dataset-metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print("Dataset prepared with", len(SELECTED_IMAGES), "images.")

