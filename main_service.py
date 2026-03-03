from animal_classifier import get_classifier
from transport_advisor import TransportAdvisor
from llm_service import FineTunedLLMService
import argparse
import os

class AnimalTransportService:
    def __init__(self, use_simple_classifier=False, model_path="./fine_tuned_animal_transport"):
        print("\n" + "="*60)
        print("ANIMAL TRANSPORT ADVISOR WITH FINE-TUNED MODEL")
        print("="*60)
        
        self.classifier = get_classifier(use_simple_classifier)
        self.advisor = TransportAdvisor()
        self.llm = FineTunedLLMService(model_path)
        
        if self.llm.model is None:
            print("\nWARNING: Fine-tuned model not loaded!")
            print("The service will use fallback responses.")
            print("To fix: run training first or check model path.\n")
        
        print("Service initialized\n")
    
    def process_request(self, image_path, origin, destination):
        print("STEP 1: Identifying animal...")
        animal, confidence = self.classifier.classify_with_fallback(image_path)
        print(f"  → Animal: {animal} (confidence: {confidence:.1%})")
        
        print("\nSTEP 2: Analyzing transport options...")
        transport_data = self.advisor.get_recommendations(animal, origin, destination)
        print(f"  → Found {transport_data['similar_shipments_found']} similar shipments")
        print(f"  → Distance: {transport_data['distance_km']} km")
        
        print("\nSTEP 3: Generating recommendation with fine-tuned model...")
        recommendation = self.llm.generate_recommendation(transport_data)
        
        return recommendation, transport_data


def main():
    parser = argparse.ArgumentParser(description="Animal Transport Advisor with Fine-tuned Model")
    parser.add_argument("--image", required=True, help="Path to animal photo")
    parser.add_argument("--origin", required=True, help="Origin city")
    parser.add_argument("--destination", required=True, help="Destination city")
    parser.add_argument("--simple", action="store_true", help="Use simple classifier")
    parser.add_argument("--model-path", default="./fine_tuned_animal_transport", 
                       help="Path to fine-tuned model")
    
    args = parser.parse_args()
    
    service = AnimalTransportService(
        use_simple_classifier=args.simple,
        model_path=args.model_path
    )
    
    recommendation, data = service.process_request(args.image, args.origin, args.destination)
    
    print("\n" + "="*60)
    print("FINAL RECOMMENDATION")
    print("="*60)
    print(recommendation)
    print("="*60)


if __name__ == "__main__":
    main()