import pandas as pd
import numpy as np

class TransportAdvisor:
    def __init__(self, dataset_path="train_data.csv"):
        self.dataset_path = dataset_path
        self.df = None
        self._load_dataset()
        
        self.transport_modes = {
            'road': 'Truck',
            'sea': 'Ship',
            'air': 'Air (cargo)',
            'air_accompanied': 'Air (with owner)',
            'rail': 'Train'
        }
        
        self.animal_weights = {
            'cow': (400, 800),
            'pig': (100, 300),
            'sheep': (40, 100),
            'goat': (40, 80),
            'horse': (500, 1000),
            'chicken': (2, 5),
            'cat': (3, 8),
            'dog': (5, 50),
            'rabbit': (1, 3),
            'fish': (0.5, 5),
            'bird': (0.1, 2)
        }
    
    def _load_dataset(self):
        try:
            self.df = pd.read_csv(self.dataset_path)
            print(f"Loaded {len(self.df)} transport records")
        except:
            print(f"Dataset not found at {self.dataset_path}")
            self.df = None
    
    def estimate_distance(self, origin, destination):
        seed = abs(hash(f"{origin}{destination}")) % 1000
        return 500 + (seed / 1000.0) * 4500
    
    def find_similar_shipments(self, animal, weight, origin, destination, limit=5):
        if self.df is None:
            return []
        
        if animal in self.animal_weights:
            min_w, max_w = self.animal_weights[animal]
            similar = self.df[
                (self.df['Weight'] >= min_w) & 
                (self.df['Weight'] <= max_w)
            ]
        else:
            self.df['weight_diff'] = abs(self.df['Weight'] - weight)
            similar = self.df.nsmallest(20, 'weight_diff')
        
        stats = []
        for mode in similar['Transport_Mode'].unique():
            mode_data = similar[similar['Transport_Mode'] == mode]
            
            avg_days = 0
            if 'Shipment_Date' in mode_data.columns and 'Actual_Arrival_Date' in mode_data.columns:
                try:
                    shipment_dates = pd.to_datetime(mode_data['Shipment_Date'])
                    arrival_dates = pd.to_datetime(mode_data['Actual_Arrival_Date'])
                    avg_days = (arrival_dates - shipment_dates).dt.days.mean()
                except:
                    avg_days = 5
            
            stats.append({
                'mode': mode,
                'count': len(mode_data),
                'avg_weight': mode_data['Weight'].mean(),
                'avg_days': avg_days,
                'avg_cost': mode_data['Declared_Value_USD'].mean() if 'Declared_Value_USD' in mode_data.columns else 0,
                'risk_flag': mode_data['Risk_Flag'].mean() if 'Risk_Flag' in mode_data.columns else 0,
                'delay_days': mode_data['Customs_Delay_Days'].mean() if 'Customs_Delay_Days' in mode_data.columns else 0
            })
        
        return sorted(stats, key=lambda x: x['count'], reverse=True)[:limit]
    
    def get_recommendations(self, animal, origin, destination, weight=None):
        if weight is None and animal in self.animal_weights:
            min_w, max_w = self.animal_weights[animal]
            weight = (min_w + max_w) / 2
        else:
            weight = 100
        
        distance = self.estimate_distance(origin, destination)
        similar = self.find_similar_shipments(animal, weight, origin, destination)
        
        if animal in ['cat', 'dog']:
            modes = ['air_accompanied', 'air', 'road']
        elif animal == 'fish':
            modes = ['sea', 'air']
        elif animal in ['cow', 'horse']:
            modes = ['sea', 'road']
        else:
            modes = ['road', 'sea', 'air']
        
        recommendations = []
        
        for mode in modes:
            mode_stats = next((s for s in similar if s['mode'] == mode), None)
            
            if mode_stats:
                days = mode_stats['avg_days']
                cost = mode_stats['avg_cost'] * (distance / 1000)
                confidence = mode_stats['count'] / 10
                risk = mode_stats['risk_flag']
                delay = mode_stats['delay_days']
            else:
                speeds = {'road': 800, 'sea': 500, 'air': 5000, 'air_accompanied': 5000}
                speed = speeds.get(mode.split('_')[0], 800)
                days = distance / speed
                if days < 1:
                    days = 1
                cost = distance * {'road': 2, 'sea': 1, 'air': 5, 'air_accompanied': 7}.get(mode.split('_')[0], 2)
                confidence = 0.5
                risk = 0.3
                delay = 1
            
            recommendations.append({
                'mode': self.transport_modes.get(mode, mode),
                'mode_code': mode,
                'days': round(days, 1),
                'cost': round(cost),
                'confidence': round(confidence * 100),
                'risk': risk,
                'delay': round(delay, 1),
                'distance': round(distance)
            })
        
        recommendations.sort(key=lambda x: x['days'])
        
        return {
            'animal': animal,
            'origin': origin,
            'destination': destination,
            'distance_km': round(distance),
            'weight_kg': round(weight, 1),
            'recommendations': recommendations,
            'similar_shipments_found': len(similar)
        }