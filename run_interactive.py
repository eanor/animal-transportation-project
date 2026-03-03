import os
from main_service import AnimalTransportService

print("="*60)
print("ANIMAL TRANSPORT ADVISOR - INTERACTIVE MODE")
print("="*60)

model_path = "./fine_tuned_animal_transport"

if not os.path.exists(model_path):
    print("Fine-tuned model not found at", model_path)
    print("The service will use fallback responses.")
    print("To use fine-tuned model, run training first.\n")
else:
    print("Found fine-tuned model at", model_path)

service = AnimalTransportService(
    use_simple_classifier=False,
    model_path=model_path
)

while True:
    print("\n" + "-"*60)
    print("MENU:")
    print("1. Get transport recommendation")
    print("2. Exit")
    
    choice = input("\nChoose (1-2): ").strip()
    
    if choice == '2':
        print("Goodbye!")
        break
    
    elif choice == '1':
        image_path = input("\nPath to animal photo: ").strip()
        
        if not os.path.exists(image_path):
            print(f"File not found: {image_path}")
            continue
        
        origin = input("Origin city: ").strip()
        destination = input("Destination city: ").strip()
        
        if not origin or not destination:
            print("Please enter both cities")
            continue
        
        print("\n" + "="*60)
        print("PROCESSING REQUEST...")
        print("="*60)
        
        recommendation, data = service.process_request(image_path, origin, destination)
        
        print("\n" + "="*60)
        print("FINAL RECOMMENDATION")
        print("="*60)
        print(recommendation)
        print("="*60)
        
        input("\nPress Enter to continue...")