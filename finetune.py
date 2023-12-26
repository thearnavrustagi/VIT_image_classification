import torch
import evaluate
import numpy as np
from datasets import load_dataset, Image
from transformers import DefaultDataCollator
from transformers import TrainingArguments, Trainer
from transformers import AutoImageProcessor, AutoModelForImageClassification
from torchvision.transforms import RandomResizedCrop, Compose, Normalize, ToTensor

CHECKPOINT = "google/vit-base-patch16-224-in21k"
# CHECKPOINT = "microsoft/beit-base-patch16-224"

MODEL_PATH = f"model_{CHECKPOINT.split('/')[-1]}"


def init_dataset(path="./dset/"):
    dataset = load_dataset(path)
    print(f"[DEBUG] {dataset}")
    return dataset


def initialise_labels(labels):
    label2id, id2label = {}, {}
    for i, label in enumerate(labels):
        label2id[label] = str(i)
        id2label[str(i)] = label

    print(f"[DEBUG] label2id : {label2id}\n[DEBUG] id2label : {id2label}")
    return label2id, id2label


def train_model(dataset, dicts):
    global CHECKPOINT
    image_processor = AutoImageProcessor.from_pretrained(CHECKPOINT)
    dataset = transform_dataset(dataset, image_processor)
    data_collator = DefaultDataCollator()

    compute_metrics = get_compute_metrics()

    label2id, id2label = dicts
    model = AutoModelForImageClassification.from_pretrained(
        CHECKPOINT,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
    )

    training_args = get_training_args()
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        tokenizer=image_processor,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    return model


def transform_dataset(dataset, image_processor):
    normalize = Normalize(
        mean=image_processor.image_mean, std=image_processor.image_std
    )
    size = (
        image_processor.size["shortest_edge"]
        if "shortest_edge" in image_processor.size
        else (image_processor.size["height"], image_processor.size["width"])
    )
    _transforms = Compose([RandomResizedCrop(size), ToTensor(), normalize])

    def transforms(examples):
        examples["pixel_values"] = [
            _transforms(img.convert("RGB")) for img in examples["image"]
        ]
        del examples["image"]
        return examples

    dataset = dataset.with_transform(transforms)
    return dataset


def get_compute_metrics():
    accuracy = evaluate.load("accuracy")

    def compute_metrics(accuracy, eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return accuracy.compute(predictions=predictions, references=labels)

    return lambda x: compute_metrics(accuracy, x)


def get_training_args():
    global MODEL_PATH
    training_args = TrainingArguments(
        output_dir=MODEL_PATH,
        remove_unused_columns=False,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=32,
        gradient_accumulation_steps=4,
        per_device_eval_batch_size=4,
        num_train_epochs=25,
        warmup_ratio=0.1,
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        push_to_hub=True,
    )

    return training_args


if __name__ == "__main__":
    dataset = init_dataset()
    label2id, id2label = initialise_labels(dataset["train"].features["label"].names)

    model = train_model(dataset, (label2id, id2label))
