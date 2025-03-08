from typing import Any, List, Union, Dict
from PIL import Image
import torch
import numpy as np
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModelForImageClassification

def load_classifier(model_name: str = "codewithdark/vit-chest-xray"):
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForImageClassification.from_pretrained(model_name)
    label_columns = ['Cardiomegaly', 'Edema', 'Consolidation', 'Pneumonia', 'No Finding']
    return processor, model, label_columns

def predict(image: Union[str, Image.Image], model, processor, label_columns: List[str]) -> Dict[str, Any]:
    if isinstance(image,str):
        image = Image.open(image)
        if image.mode != 'RGB':
            image = image.convert('RGB')

    inputs = processor(images=image, return_tensors='pt')
    
    with torch.no_grad():
        output = model(**inputs)
    logits = output.logits
    probabilities = F.softmax(logits, dim=-1)[0]
    predicted_class_idx = torch.argmax(logits, dim=-1).item()
    predicted_class_label = label_columns[predicted_class_idx]

    results = {
        "logits": logits,
        "probabilities": probabilities,
        "predicted_class_idx": predicted_class_idx,
        "predicted_class_label": predicted_class_label,
    }
    return results
