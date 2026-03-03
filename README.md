# Animal Transport Advisor

A machine learning system that provides transport recommendations for animals based on photographs and route information. The system combines computer vision for species identification with a fine-tuned language model (TinyLlama-1.1B) trained on shipment data from , that was augmented for the task specifics (added random weight for cargo from 1kg to 1000kg and added random cities for countries data for more specified shipment details), and animal welfare guidelines.

## Overview

This project implements an end-to-end solution for animal transport consultation:
- **Animal identification** from photographs using ResNet50
- **Transport mode recommendations** based on partially modded shipment pattern data
- **Cost and duration estimates** derived from similar past shipments
- **Welfare considerations** integrated from EU regulations and literature on animal welfare

## Files Description

| File | Description |
|------|-------------|
| `shipment_service.ipynb` | Main Jupyter notebook for Colab - contains all code and interactive interface |
| `animal_classifier.py` | ResNet50-based classifier for identifying animal species from images |
| `transport_advisor.py` | Analyzes historical shipment data to find similar transports and calculate statistics |
| `welfare_guidelines.py` | Database of animal welfare requirements from EU Regulation 1/2005 and scientific studies |
| `llm_service.py` | Fine-tuned TinyLlama model for generating natural language recommendations |
| `main_service.py` | Orchestrates all components: classification, analysis and recommendation generation |
| `run_interactive.py` | Interactive command-line interface for testing |
| `prepare_training_data.py` | Converts raw shipment CSV into training examples for LLM fine-tuning |
| `train_llm.py` | Fine-tunes TinyLlama on the prepared dataset using LoRA |
| `requirements.txt` | Python dependencies |

## Dataset Format

The system expects a CSV file (`train_data.csv`) with the following columns:
- `Shipment_ID`, `Origin_Country`, `Destination_Country`, `Shipment_Date`, `Actual_Arrival_Date`
- `Transport_Mode`, `Carrier_Name`, `Declared_Value_USD`, `Tariff_Category`
- `Route_Risk_Index`, `Inspection_Type`, `Delay_Reason`, `Customs_Delay_Days`
- `Risk_Flag`, `Origin_City`, `Destination_City`, `Weight`

## Quick Start in Google Colab

1. Open `shipment_service.ipynb` in Google Colab
2. Upload your `train_data.csv` file
3. Run all cells in order:
   - Cell 1: Install dependencies
   - Cell 2: Prepare training data
   - Cell 3: Train the model (or skip if using pre-trained)
   - Cell 4: Run interactive service

```python
# Alternative: Command line usage
python main_service.py --image cow.jpg --origin "Sydney" --destination "Mumbai"
```

## Interface example

```java
MENU:
1. Get transport recommendation
2. Exit

Choose (1-2): 1

Path to animal photo: cow_01.jpg

Origin city: Sydney

Destination city: Mumbai

============================================================

PROCESSING REQUEST...

============================================================

STEP 1: Identifying animal...

  → Animal: cow (confidence: 96.2%)

STEP 2: Analyzing transport options...

  → Found 4 similar shipments
  
  → Distance: 1949 km

STEP 3: Generating recommendation with fine-tuned model...

============================================================

FINAL RECOMMENDATION

============================================================

Based on historical shipment data and animal welfare guidelines, recommended best transport option: Sea transport. Expected duration and cost breakdown: 3.9 days at cost $1949. Potential risks and mitigation strategies: avoid mixing unfamiliar animals. Additional notes: Stress increases when mixed with unfamiliar animals. Mortality increases beyond 100km. Recommendations:

1. Seek transport recommendation from professional transport agent
2. Ensure proper documentation including health certificates and customs declarations
3. Schedule transport for when expected increase in risk is below 100km
4. Monitor progress closely for any issues or delays
5. Use customs brokerage services for easy entry and exit
6. Ensure proper hygiene and cleanliness in shipping container
7. Follow recommended animal welfare guidelines
8. Avoid mixing unfamiliar animals.

============================================================

Press Enter to continue...

------------------------------------------------------------
```
