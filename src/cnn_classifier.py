"""
CNN-based Quantum Circuit Classifier for the Quantum Circuit Dataset Pipeline.

This module uses a fine-tuned ConvNeXt Tiny model to classify extracted figures
as quantum circuits or non-circuits. The model was trained on labeled examples
of quantum circuit diagrams.

Author: [Your Name]
Exam ID: 37
"""

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from pathlib import Path
from typing import Dict, Optional
from dataclasses import dataclass

import sys
sys.path.append(str(Path(__file__).parent))

from utils.logging_utils import setup_logger

# Module logger
logger = setup_logger(__name__)

@dataclass
class CNNClassificationResult:
    """
    Result of CNN-based circuit classification.
    
    Attributes
    ----------
    is_circuit : bool
        Whether the figure is classified as a quantum circuit.
    confidence : float
        Confidence score (0-1) from the model.
    class_label : str
        Predicted class label ('circuit' or 'non_circuit').
    probabilities : Dict[str, float]
        Probabilities for each class.
    """
    is_circuit: bool
    confidence: float
    class_label: str
    probabilities: Dict[str, float]


class CNNCircuitClassifier:
    """
    CNN-based classifier using fine-tuned ConvNeXt Tiny model.
    
    This classifier uses a pre-trained ConvNeXt Tiny model that has been
    fine-tuned on quantum circuit images to distinguish between circuit
    diagrams and other types of figures.
    
    Attributes
    ----------
    model : nn.Module
        The loaded ConvNeXt Tiny model.
    device : torch.device
        Device to run inference on (GPU/CPU).
    transform : transforms.Compose
        Image preprocessing pipeline.
    class_names : list
        List of class names ['circuit', 'non_circuit'].
    threshold : float
        Confidence threshold for circuit classification.
    
    Examples
    --------
    >>> classifier = CNNCircuitClassifier()
    >>> result = classifier.classify("path/to/figure.png")
    >>> print(f"Is circuit: {result.is_circuit}, Confidence: {result.confidence:.2f}")
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        threshold: float = 0.5,
        device: Optional[str] = None
    ):
        """
        Initialize the CNN classifier.
        
        Parameters
        ----------
        model_path : str, optional
            Path to the saved model weights (.pth file).
            If None, searches in default locations.
        threshold : float
            Confidence threshold for classifying as circuit (default: 0.5).
        device : str, optional
            Device to use ('cuda' or 'cpu'). Auto-detected if None.
        """
        self.threshold = threshold
        
        # Setup device
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"CNN Classifier using device: {self.device}")
        
        # Class names (must match training order from ImageFolder)
        # ImageFolder sorts alphabetically: 'circuit' comes before 'non_circuit'
        self.class_names = ['circuit', 'non_circuit']
        
        # Image preprocessing (must match training transforms)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # Load the model
        self.model = self._load_model(model_path)
        self.model.eval()
        
        logger.info("CNN Circuit Classifier initialized successfully")
    
    def _load_model(self, model_path: Optional[str] = None) -> nn.Module:
        """
        Load the fine-tuned ConvNeXt Tiny model.
        
        Parameters
        ----------
        model_path : str, optional
            Path to model weights. Searches default locations if None.
        
        Returns
        -------
        nn.Module
            Loaded model ready for inference.
        """
        # Find model file
        if model_path:
            path = Path(model_path)
        else:
            # Search in default locations
            base_dir = Path(__file__).parent.parent
            search_paths = [
                base_dir / "circuit_classifier.pth",
                base_dir / "src" / "circuit_classifier.pth",
                base_dir / "models" / "circuit_classifier.pth",
            ]
            
            path = None
            for p in search_paths:
                if p.exists():
                    path = p
                    break
            
            if path is None:
                raise FileNotFoundError(
                    f"Model file 'circuit_classifier.pth' not found. "
                    f"Searched in: {[str(p) for p in search_paths]}"
                )
        
        logger.info(f"Loading model from: {path}")
        
        # Initialize ConvNeXt Tiny architecture
        model = models.convnext_tiny(weights=None)  # Don't load pretrained weights
        
        # Modify classifier for 2 classes (must match training)
        num_ftrs = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(num_ftrs, 2)
        
        # Load saved weights
        state_dict = torch.load(path, map_location=self.device)
        model.load_state_dict(state_dict)
        
        # Move to device
        model = model.to(self.device)
        
        return model
    
    def classify(self, image_path: str) -> CNNClassificationResult:
        """
        Classify a single image as circuit or non-circuit.
        
        Parameters
        ----------
        image_path : str
            Path to the image file to classify.
        
        Returns
        -------
        CNNClassificationResult
            Classification result with confidence and probabilities.
        """
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Run inference
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.softmax(outputs, dim=1)[0]
                
                # Get prediction
                circuit_prob = probabilities[0].item()  # 'circuit' class
                non_circuit_prob = probabilities[1].item()  # 'non_circuit' class
            
            # Determine classification
            is_circuit = circuit_prob >= self.threshold
            predicted_class = 'circuit' if is_circuit else 'non_circuit'
            confidence = circuit_prob if is_circuit else non_circuit_prob
            
            return CNNClassificationResult(
                is_circuit=is_circuit,
                confidence=confidence,
                class_label=predicted_class,
                probabilities={
                    'circuit': circuit_prob,
                    'non_circuit': non_circuit_prob
                }
            )
            
        except Exception as e:
            logger.error(f"Error classifying image {image_path}: {e}")
            # Return negative result on error
            return CNNClassificationResult(
                is_circuit=False,
                confidence=0.0,
                class_label='error',
                probabilities={'circuit': 0.0, 'non_circuit': 0.0}
            )
    
    def analyze(self, image_path: str) -> Dict:
        """
        Analyze an image for circuit detection (compatible with RobustRasterDetector interface).
        
        This method provides compatibility with the existing pipeline interface.
        
        Parameters
        ----------
        image_path : str
            Path to the image file.
        
        Returns
        -------
        Dict
            Analysis result with 'score', 'reason', and 'features' keys.
        """
        result = self.classify(image_path)
        
        # Convert to score (0-1 where higher = more likely circuit)
        score = result.probabilities['circuit']
        
        if result.is_circuit:
            reason = f"CNN: Circuit detected (confidence: {result.confidence:.2%})"
        else:
            reason = f"CNN: Non-circuit (confidence: {result.confidence:.2%})"
        
        return {
            'score': score,
            'reason': reason,
            'features': {
                'is_circuit': result.is_circuit,
                'circuit_probability': result.probabilities['circuit'],
                'non_circuit_probability': result.probabilities['non_circuit'],
                'class_label': result.class_label
            }
        }


# Convenience function for quick classification
def classify_circuit_image(
    image_path: str,
    model_path: Optional[str] = None,
    threshold: float = 0.5
) -> CNNClassificationResult:
    """
    Quick utility function to classify a single image.
    
    Parameters
    ----------
    image_path : str
        Path to the image file.
    model_path : str, optional
        Path to model weights.
    threshold : float
        Classification threshold.
    
    Returns
    -------
    CNNClassificationResult
        Classification result.
    """
    classifier = CNNCircuitClassifier(model_path=model_path, threshold=threshold)
    return classifier.classify(image_path)


if __name__ == "__main__":
    # Test the classifier
    import sys
    
    if len(sys.argv) > 1:
        test_image = sys.argv[1]
        classifier = CNNCircuitClassifier()
        result = classifier.classify(test_image)
        
        print(f"\nClassification Result for: {test_image}")
        print(f"  Is Circuit: {result.is_circuit}")
        print(f"  Class: {result.class_label}")
        print(f"  Confidence: {result.confidence:.2%}")
        print(f"  Probabilities: {result.probabilities}")
    else:
        print("Usage: python cnn_classifier.py <image_path>")
        print("\nInitializing classifier to verify model loading...")
        try:
            classifier = CNNCircuitClassifier()
            print("✅ Classifier initialized successfully!")
        except Exception as e:
            print(f"❌ Error: {e}")
