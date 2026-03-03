welfare_guidelines = {
    'cow': {
        'max_hours': 8,
        'min_rest': 1,
        'requirements': [
            'ventilation',
            'water access',
            'rest stops every 8 hours',
            'avoid mixing unfamiliar animals'
        ],
        'notes': 'Stress increases when mixed with unfamiliar animals. Mortality increases beyond 100km.'
    },
    'pig': {
        'max_hours': 8,
        'min_rest': 1,
        'requirements': [
            'temperature control',
            'ventilation',
            'water access',
            'avoid overcrowding'
        ],
        'notes': 'Susceptible to heat stress. Short journeys can be more stressful than long ones.'
    },
    'sheep': {
        'max_hours': 8,
        'min_rest': 0.5,
        'requirements': [
            'ventilation',
            'careful loading',
            'avoid sudden movements'
        ],
        'notes': 'More resilient than cattle but require careful handling during loading.'
    },
    'goat': {
        'max_hours': 8,
        'min_rest': 0.5,
        'requirements': [
            'ventilation',
            'secure partitions'
        ],
        'notes': 'Similar to sheep but more agile.'
    },
    'horse': {
        'max_hours': 6,
        'min_rest': 2,
        'requirements': [
            'specialized vehicle',
            'attendant',
            'frequent rest',
            'individual stalls'
        ],
        'notes': 'Require specialized horse transport vehicles and frequent breaks.'
    },
    'chicken': {
        'max_hours': 12,
        'min_rest': 0.5,
        'requirements': [
            'ventilation',
            'temperature control',
            'proper crate density'
        ],
        'notes': 'Vulnerable to temperature extremes. Crate design crucial for welfare.'
    },
    'cat': {
        'max_hours': 6,
        'min_rest': 1,
        'requirements': [
            'soft carrier',
            'temperature control',
            'minimize noise'
        ],
        'notes': 'Can travel in cabin with owner. Stress from unfamiliar environment.'
    },
    'dog': {
        'max_hours': 8,
        'min_rest': 1,
        'requirements': [
            'carrier or harness',
            'walk breaks',
            'food and water',
            'temperature control'
        ],
        'notes': 'Small dogs can travel in cabin. Larger dogs require cargo with special handling.'
    },
    'rabbit': {
        'max_hours': 6,
        'min_rest': 1,
        'requirements': [
            'temperature control',
            'quiet environment',
            'hay and water'
        ],
        'notes': 'Highly sensitive to temperature and stress.'
    },
    'fish': {
        'max_hours': 24,
        'min_rest': 0,
        'requirements': [
            'oxygenated water',
            'temperature control',
            'specialized tanks',
            'minimal movement'
        ],
        'notes': 'Require specialized aquariums with aeration. Water quality critical.'
    },
    'bird': {
        'max_hours': 12,
        'min_rest': 0.5,
        'requirements': [
            'ventilation',
            'darkened containers',
            'temperature control'
        ],
        'notes': 'Sensitive to drafts and temperature changes.'
    }
}

def get_welfare_guidelines(animal):
    return welfare_guidelines.get(animal, welfare_guidelines['chicken'])