# llm_service.py - ИСПРАВЛЕННАЯ ВЕРСИЯ
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
from welfare_guidelines import get_welfare_guidelines

class FineTunedLLMService:
    def __init__(self, model_path="./fine_tuned_animal_transport"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_path = model_path
        
        print(f"Loading fine-tuned model from {model_path}...")
        
        if not os.path.exists(model_path):
            print(f"Fine-tuned model not found at {model_path}")
            self.model = None
            self.tokenizer = None
            return
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            ).to(self.device)
            
            print(f"Fine-tuned model loaded successfully on {self.device}")
            
        except Exception as e:
            print(f"Error loading fine-tuned model: {e}")
            self.model = None
            self.tokenizer = None
    
    def generate_recommendation(self, transport_data):
        if self.model is None:
            return self._generate_fallback(transport_data)
        
        prompt = self._create_prompt(transport_data)
        
        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=400,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        if prompt in full_response:
            response = full_response[len(prompt):].strip()
        else:
            response = full_response
        
        return response
    
    def _create_prompt(self, data):
        animal = data['animal']
        origin = data['origin']
        destination = data['destination']
        distance = data['distance_km']
        weight = data['weight_kg']
        
        welfare = get_welfare_guidelines(animal)
        
        prompt = f"""User: How to transport a {animal} from {origin} to {destination}?

Transport options from historical data:
"""
        
        for i, rec in enumerate(data['recommendations'], 1):
            prompt += f"{i}. {rec['mode']}: {rec['days']} days, ${rec['cost']}, confidence {rec['confidence']}%\n"
        
        prompt += f"""
Animal welfare guidelines for {animal}:
- Maximum journey time: {welfare['max_hours']} hours
- Minimum rest: {welfare['min_rest']} hours
- Requirements: {', '.join(welfare['requirements'])}
- Additional notes: {welfare['notes']}

Based on historical shipment data and animal welfare guidelines, provide detailed recommendations including:
1. Best transport option considering time, cost and animal welfare
2. Expected duration and cost breakdown
3. Special requirements for this animal type
4. Potential risks and mitigation strategies
5. References to EU Regulation 1/2005 and scientific studies

Assistant:"""
        
        return prompt
    
    def _generate_fallback(self, data):
        best = data['recommendations'][0]
        welfare = get_welfare_guidelines(data['animal'])
        
        response = f"""
Based on analysis of {data['similar_shipments_found']} similar shipments:

RECOMMENDED TRANSPORT: {best['mode']}
- Travel time: {best['days']} days
- Estimated cost: ${best['cost']}
- Distance: {data['distance_km']} km
- Confidence in recommendation: {best['confidence']}%

ALTERNATIVE OPTIONS:
"""
        for rec in data['recommendations'][1:3]:
            response += f"- {rec['mode']}: {rec['days']} days, ${rec['cost']}\n"
        
        response += f"""
ANIMAL WELFARE GUIDELINES FOR {data['animal'].upper()}:
- Maximum recommended journey time: {welfare['max_hours']} hours
- Required rest stops: every {welfare['min_rest']} hours
- Special requirements: {', '.join(welfare['requirements'])}
- {welfare['notes']}

According to EU Regulation 1/2005, animals must be fit for transport and vehicles must meet specific standards for ventilation, temperature, and space. Long journeys require additional planning and documentation.

POTENTIAL RISKS:
- Customs delay probability: {best['delay']} days average
- Risk index: {'High' if best['risk'] > 0.5 else 'Medium' if best['risk'] > 0.2 else 'Low'}
- Main risk factors: incomplete paperwork, health inspections, weather conditions

RECOMMENDATIONS:
1. Book with certified animal transport carrier
2. Prepare veterinary health certificate and export documents
3. Ensure proper crate/container meeting IATA or EU standards
4. Plan for rest stops and food/water according to guidelines
5. Monitor weather conditions along the route
"""
        
        return response