# prepare_training_data.py
import pandas as pd
import json
import os
from datetime import datetime

print("Loading dataset...")
df = pd.read_csv('train_data.csv')
print(f"Loaded {len(df)} records")

def determine_animal(weight):
    if weight < 5:
        return "chicken"
    elif 5 <= weight < 20:
        return "cat"
    elif 20 <= weight < 50:
        return "dog"
    elif 50 <= weight < 100:
        return "sheep"
    elif 100 <= weight < 300:
        return "pig"
    elif 300 <= weight < 600:
        return "cow"
    elif 600 <= weight < 1000:
        return "horse"
    else:
        return "elephant"

def calculate_days(row):
    try:
        shipment = pd.to_datetime(row['Shipment_Date'])
        arrival = pd.to_datetime(row['Actual_Arrival_Date'])
        return (arrival - shipment).days
    except:
        return 5

def create_prompt(row):
    animal = determine_animal(row['Weight'])
    days = calculate_days(row)
    
    risk_text = "high risk" if row['Risk_Flag'] == 1 else "low risk"
    delay_text = f"delayed {row['Customs_Delay_Days']} days: {row['Delay_Reason']}" if row['Customs_Delay_Days'] > 0 else "no delays"
    
    prompt = f"How to transport a {animal} from {row['Origin_City']} to {row['Destination_City']}?"
    
    completion = f"""Based on historical shipment data:

Transport mode: {row['Transport_Mode']}
Travel time: {days} days
Cost: ${row['Declared_Value_USD']}
Carrier: {row['Carrier_Name']}
Risk level: {risk_text}
Customs: {delay_text}
Route risk index: {row['Route_Risk_Index']}

Recommendation: Use {row['Transport_Mode']} transport. Expected duration {days} days at cost ${row['Declared_Value_USD']}. {risk_text.capitalize()} route with {delay_text}. Ensure proper documentation including health certificates and customs declarations."""
    
    return {
        'prompt': prompt,
        'completion': completion
    }

print("Creating training data...")
training_data = []

for idx, row in df.iterrows():
    training_data.append(create_prompt(row))
    
    if (idx + 1) % 1000 == 0:
        print(f"Processed {idx + 1}/{len(df)} records")

with open('training_data.jsonl', 'w') as f:
    for item in training_data:
        f.write(json.dumps(item) + '\n')

print(f"Created {len(training_data)} training examples")
print("Sample prompt:", training_data[0]['prompt'])
print("Sample completion:", training_data[0]['completion'][:100], "...")