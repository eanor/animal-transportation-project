import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import requests
from io import BytesIO
import os

class AnimalClassifier:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading classifier on {self.device}...")
        
        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        self.labels = self._load_labels()
        
        self.animal_mapping = {
            'cow': ['cow', 'ox', 'bull', 'cattle', 'bovine'],
            'pig': ['pig', 'hog', 'boar', 'sow'],
            'sheep': ['sheep', 'lamb', 'ram', 'ewe'],
            'goat': ['goat'],
            'horse': ['horse', 'pony', 'mare', 'stallion'],
            'chicken': ['chicken', 'hen', 'rooster', 'cock'],
            'cat': ['cat', 'kitty', 'kitten', 'tabby'],
            'dog': ['dog', 'puppy', 'pup', 'canine'],
            'rabbit': ['rabbit', 'bunny', 'hare'],
            'fish': ['fish', 'goldfish', 'koi', 'tuna', 'salmon'],
            'bird': ['bird', 'pigeon', 'sparrow', 'eagle', 'hawk']
        }
        
        print("Classifier loaded successfully")
    
    def _load_labels(self):
        try:
            url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
            response = requests.get(url, timeout=5)
            return response.json()
        except:
            return ["unknown"] * 1000
    
    def classify(self, image_path):
        try:
            if image_path.startswith(('http://', 'https://')):
                response = requests.get(image_path, timeout=10)
                image = Image.open(BytesIO(response.content)).convert('RGB')
            else:
                if not os.path.exists(image_path):
                    return "unknown", 0.0
                image = Image.open(image_path).convert('RGB')
            
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            
            top5_prob, top5_indices = torch.topk(probabilities, 5)
            
            for prob, idx in zip(top5_prob, top5_indices):
                if idx < len(self.labels):
                    label = self.labels[idx].lower()
                    
                    for animal, keywords in self.animal_mapping.items():
                        if any(keyword in label for keyword in keywords):
                            return animal, prob.item()
            
            return "unknown", 0.0
            
        except Exception as e:
            print(f"Classification error: {e}")
            return "unknown", 0.0
    
    def classify_with_fallback(self, image_path):
        animal, confidence = self.classify(image_path)
        
        if animal == "unknown" or confidence < 0.3:
            print(f"Could not identify animal with confidence (got: {animal}, {confidence:.1%})")
            print("Available animals: cow, pig, sheep, goat, horse, chicken, cat, dog, rabbit, fish, bird")
            animal = input("Please enter animal type manually: ").strip().lower()
            confidence = 1.0
            
        return animal, confidence


class SimpleClassifier:
    def classify(self, image_path):
        filename = os.path.basename(image_path).lower()
        
        keywords = {
            'cow': ['cow', 'bull', 'ox', 'calf'],
            'pig': ['pig', 'hog', 'swine'],
            'sheep': ['sheep', 'lamb'],
            'goat': ['goat'],
            'horse': ['horse', 'pony', 'mare'],
            'chicken': ['chicken', 'hen', 'rooster'],
            'cat': ['cat', 'kitten'],
            'dog': ['dog', 'puppy'],
            'fish': ['fish'],
            'bird': ['bird']
        }
        
        for animal, words in keywords.items():
            if any(word in filename for word in words):
                return animal, 0.8
        
        return "unknown", 0.0


def get_classifier(use_simple=False):
    if use_simple:
        return SimpleClassifier()
    else:
        try:
            return AnimalClassifier()
        except Exception as e:
            print(f"Error loading main classifier: {e}")
            print("Switching to simple classifier")
            return SimpleClassifier()