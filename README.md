# Car Recommendation Engine

A Python-based car recommendation system built around Jupyter notebook functionality that imports machine learning models from `notebooks/rec_engine.ipynb` and uses automotive data from `data/data.csv`.

## Project Structure

```
project/
├── car_recommendation_system.py          # Main recommendation engine class
├── car_recommendation_demo.ipynb         # Primary Jupyter notebook interface
├── run_car_recommendations.py            # Optional command-line interface
├── data/
│   ├── data.csv                         # Primary automotive dataset (11,914 cars)
│   └── vehicles_dataset.csv             # Additional automotive dataset (optional)
└── notebooks/
    └── rec_engine.ipynb                 # Original notebook with ML models
```

## Features

- **Jupyter Notebook Integration**: Primary interface through interactive notebooks with step-by-step examples
- **Exact Notebook Compatibility**: Uses the same `get_recommendations_by_preference` function signature as the original notebook
- **Advanced ML Pipeline**: Random Forest-based recommendation algorithm with comprehensive preprocessing
- **Interactive Widgets**: Built-in ipywidgets support for dynamic preference selection
- **Multiple Dataset Support**: Combines data from multiple CSV sources automatically
- **Flexible Preferences**: Support for both categorical and numerical car features
- **Data Export**: Save recommendations and analysis to CSV files

## Getting Started with Jupyter Notebooks

### 1. Primary Interface - Jupyter Notebook Demo

Open `car_recommendation_demo.ipynb` for the complete interactive experience:

```python
# Import the car recommendation system
from car_recommendation_system import CarRecommendationEngine, create_recommendation_system, run_example_recommendations

# Quick start - run comprehensive examples
rec_engine = run_example_recommendations('data/data.csv')
```

This will automatically:
- Load and preprocess both datasets (`data.csv` and `vehicles_dataset.csv`)
- Display dataset statistics and sample cars
- Run example recommendations for different buyer types (budget, performance, luxury)
- Show the exact function signatures for custom usage

### 2. Manual Notebook Setup

For custom analysis and recommendations:

```python
# Initialize the recommendation engine
rec_engine = create_recommendation_system('data/data.csv', 'data/vehicles_dataset.csv')

# Explore the dataset
rec_engine.get_feature_info()          # Show all available features and ranges
rec_engine.display_sample_data(5)      # Display sample cars from dataset

# Get recommendations using exact notebook function signature
user_preferences = {
    'Vehicle Style': '4dr Hatchback',
    'Vehicle Size': 'Compact',
    'Engine HP': 100,
    'Year': 2015,
    'Transmission Type': 'AUTOMATIC',
    'Market Category': 'Hatchback,Hybrid'
}

recommendations = rec_engine.get_recommendations_by_preference(
    user_preferences,
    rec_engine.preprocessor,
    rec_engine.feature_matrix,
    rec_engine.df_rec,
    top_n=10
)
```

### 3. Interactive Widget Interface

The notebook includes an interactive widget interface (requires ipywidgets):

```python
# Create interactive sliders and dropdowns for all features
rec_engine.create_interactive_interface()
```

This provides:
- Dynamic sliders for numerical features (Year, Engine HP, MSRP, etc.)
- Dropdown menus for categorical features (Vehicle Size, Market Category, etc.)
- Real-time recommendation updates as you adjust preferences
- Integrated display of results within the notebook

### 4. Notebook Analysis Features

The Jupyter environment provides additional analysis capabilities:

```python
# Dataset exploration
print(f"Total cars in dataset: {len(rec_engine.df_rec)}")
print(f"Feature matrix shape: {rec_engine.feature_matrix.shape}")

# Get unique values for categorical features
vehicle_sizes = rec_engine.get_unique_values('Vehicle Size')
market_categories = rec_engine.get_unique_values('Market Category')

# Custom filtering and analysis
budget_cars = rec_engine.df_rec[rec_engine.df_rec['MSRP'] <= 25000]
performance_cars = rec_engine.df_rec[rec_engine.df_rec['Engine HP'] >= 300]
```

## Recommendation Examples in Notebooks

### Budget-Conscious Buyer
```python
budget_preferences = {
    'MSRP': 25000,              # Maximum budget
    'highway MPG': 30,          # Minimum fuel efficiency
    'Vehicle Size': 'Compact',  # Smaller, affordable cars
    'Transmission Type': 'AUTOMATIC'
}

budget_recs = rec_engine.get_recommendations_by_preference(budget_preferences, top_n=10)
```

### Performance Enthusiast  
```python
performance_preferences = {
    'Engine HP': 250,           # High horsepower requirement
    'Market Category': 'Performance',
    'Transmission Type': 'MANUAL',
    'Vehicle Style': 'Coupe'
}

performance_recs = rec_engine.get_recommendations_by_preference(performance_preferences, top_n=10)
```

### Family Buyer
```python
family_preferences = {
    'Vehicle Size': 'Midsize',  # More interior space
    'Number of Doors': 4,       # Four doors required
    'MSRP': 35000,             # Family budget range
    'Vehicle Style': 'SUV'     # Family-friendly vehicle type
}

family_recs = rec_engine.get_recommendations_by_preference(family_preferences, top_n=10)
```

### Eco-Friendly Buyer
```python
eco_preferences = {
    'highway MPG': 40,          # High fuel efficiency
    'Market Category': 'Hybrid',
    'Engine Fuel Type': 'hybrid'
}

eco_recs = rec_engine.get_recommendations_by_preference(eco_preferences, top_n=10)
```

## Available Features for Analysis

### Numerical Features (with ranges)
- **Year**: 1990-2017 (vehicle model year)
- **Engine HP**: 55-1001 horsepower
- **Engine Cylinders**: 0-16 cylinders
- **highway MPG**: 12-354 miles per gallon
- **city mpg**: 7-137 miles per gallon
- **MSRP**: $2,000-$2,065,902 (manufacturer suggested retail price)
- **Number of Doors**: 2-4 doors
- **Age**: Calculated as 2025 - Year
- **Price_per_HP**: Price per horsepower ratio
- **Efficiency**: Custom efficiency metric (MPG/HP * 100)

### Categorical Features (with examples)
- **Market Category**: Luxury, Performance, Compact, Hybrid, Hatchback, Convertible, etc.
- **Vehicle Size**: Compact, Midsize, Large
- **Vehicle Style**: Sedan, SUV, Hatchback, Coupe, Convertible, Pickup, etc.
- **Transmission Type**: AUTOMATIC, MANUAL, AUTOMATED_MANUAL, DIRECT_DRIVE
- **Driven_Wheels**: front wheel drive, rear wheel drive, all wheel drive, four wheel drive
- **Engine Fuel Type**: regular unleaded, premium unleaded, hybrid, electric, diesel, flex-fuel, etc.

## Algorithm Implementation

The recommendation system implements a sophisticated machine learning pipeline:

1. **Data Preprocessing**: 
   - Handles missing values using median imputation (numerical) and mode imputation (categorical)
   - Creates derived features: Age, Price_per_HP, Efficiency, price_to_efficiency
   - One-hot encoding for categorical features with unknown value handling
   - MinMax scaling for numerical features

2. **Preference Filtering**: 
   - Numerical features: Greater than or equal to threshold filtering (except MSRP which uses less than or equal)
   - Categorical features: Case-insensitive substring matching for flexible matching

3. **Similarity Calculation**:
   - Trains Random Forest Regressor on filtered dataset
   - Uses price_to_efficiency as target variable for learning user preferences
   - Calculates similarity scores as negative absolute difference from predicted user score
   - Returns top N matches sorted by highest similarity scores

## Dataset Integration

The system automatically processes and combines multiple data sources:

### Primary Dataset (data.csv)
- 11,914 vehicles with complete feature sets
- Professional automotive data with consistent formatting
- Comprehensive market category and technical specifications

### Secondary Dataset (vehicles_dataset.csv)
- Additional 1,000+ vehicles with parsed engine specifications
- Automatic standardization to match primary dataset format
- Engine specification parsing from text descriptions
- Fuel type and transmission standardization

## Requirements for Notebook Usage

```
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
jupyter>=1.0.0
ipywidgets>=7.6.0    # Required for interactive widgets
```

## Installation and Setup

1. Clone or download the project files
2. Ensure data files are in the correct locations:
   - `data/data.csv` (required)
   - `data/vehicles_dataset.csv` (optional, for expanded dataset)
3. Install required packages:
   ```bash
   pip install pandas numpy scikit-learn matplotlib jupyter ipywidgets
   ```
4. Launch Jupyter notebook:
   ```bash
   jupyter notebook car_recommendation_demo.ipynb
   ```

## Data Format Requirements

The CSV files should contain the following columns:

**Required Columns:**
- Make, Model, Year
- Engine HP, Engine Cylinders, Engine Fuel Type  
- Transmission Type, Driven_Wheels, Number of Doors
- Market Category, Vehicle Size, Vehicle Style
- highway MPG, city mpg, Popularity, MSRP

The system automatically handles missing values and creates derived features. For the secondary dataset, engine specifications can be provided as text descriptions that will be automatically parsed.

## Troubleshooting

### Jupyter Notebook Issues

**"Data file not found"**
- Verify `data/data.csv` exists in the project directory
- Check file paths in notebook cells
- Ensure proper file permissions

**"ImportError: ipywidgets"**
- Install ipywidgets: `pip install ipywidgets`
- Enable jupyter extension: `jupyter nbextension enable --py widgetsnbextension`
- Restart Jupyter notebook after installation

**"No recommendations found"**
- Use `rec_engine.get_unique_values('column_name')` to check available options
- Try relaxing preference criteria (lower thresholds for numerical features)
- Verify categorical values match dataset format using `rec_engine.get_feature_info()`

**Interactive widgets not displaying**
- Ensure ipywidgets is properly installed and enabled
- Try restarting Jupyter kernel
- Check browser compatibility with Jupyter widgets

## Alternative Usage

For users who prefer command-line interaction, a simple interface is available:

```bash
python run_car_recommendations.py
```

However, the Jupyter notebook interface provides the most comprehensive and interactive experience for exploring the recommendation system.

## License

This project extends the automotive dataset and machine learning models from `notebooks/rec_engine.ipynb` with enhanced preprocessing, interactive features, and multi-dataset integration capabilities.