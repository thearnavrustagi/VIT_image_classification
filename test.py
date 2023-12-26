from transformers import AutoImageProcessor, ViTForImageClassification
import torch
from datasets import load_dataset

dataset = load_dataset("raw_dset")
img_and_labels = zip(dataset["test"]["image"], dataset["test"]["label"])

image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
model = ViTForImageClassification.from_pretrained("./model/final")

for image, label in img_and_labels:
    inputs = image_processor(image, return_tensors="pt")

    with torch.no_grad():
        logits = model(**inputs).logits

    # model predicts one of the 1000 ImageNet classes
    predicted_label = logits.argmax(-1).item()
    print("Trying new image")
    print(f"[ TRUTH ] {model.config.id2label[label]}")
    print(f"[ PRED  ] {model.config.id2label[predicted_label]}")
