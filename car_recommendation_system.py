import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
import re
try:
    import ipywidgets as widgets
    from ipywidgets import interact, interactive, fixed, interact_manual
    from IPython.display import display, clear_output
    WIDGETS_AVAILABLE = True
except ImportError:
    WIDGETS_AVAILABLE = False

warnings.filterwarnings('ignore')

class CarRecommendationEngine:
    def __init__(self, data_csv_path='data.csv', vehicles_csv_path='vehicles_dataset.csv'):
        """
        Initialize the Car Recommendation Engine

        Parameters:
        data_csv_path (str): Path to the original data.csv file
        vehicles_csv_path (str): Path to the vehicles_dataset.csv file
        """
        self.data_csv_path = data_csv_path
        self.vehicles_csv_path = vehicles_csv_path
        self.df_original = None
        self.df_rec = None
        self.preprocessor = None
        self.feature_matrix = None

        self.numerical_features = [
            'Year', 'Engine HP', 'Engine Cylinders', 'highway MPG', 'city mpg',
            'Efficiency', 'Age', 'MSRP', 'Number of Doors', 'Price_per_HP'
        ]

        self.categorical_features = [
            'Market Category', 'Vehicle Size', 'Vehicle Style', 'Transmission Type',
            'Driven_Wheels', 'Engine Fuel Type'
        ]

    def parse_engine_hp(self, engine_str):
        """Extract horsepower from engine description"""
        if pd.isna(engine_str) or engine_str == '':
            return None

        engine_str = str(engine_str).upper()

        # Look for HP patterns in the string
        hp_patterns = [
            r'(\d+)\s*HP',
            r'(\d+)\s*HORSEPOWER', 
            r'HP\s*(\d+)',
            r'HORSEPOWER\s*(\d+)'
        ]

        for pattern in hp_patterns:
            match = re.search(pattern, engine_str)
            if match:
                return int(match.group(1))

        # If no HP found, estimate based on engine size and type
        # Look for engine displacement
        displacement_match = re.search(r'(\d+\.?\d*)\s*L', engine_str)
        if displacement_match:
            displacement = float(displacement_match.group(1))

            # Rough HP estimation based on displacement and engine type
            if 'TURBO' in engine_str or 'SUPERCHARG' in engine_str:
                hp_estimate = int(displacement * 80)  # Turbocharged engines
            elif 'V8' in engine_str or '8' in engine_str:
                hp_estimate = int(displacement * 70)  # V8 engines
            elif 'V6' in engine_str or '6' in engine_str:
                hp_estimate = int(displacement * 60)  # V6 engines
            elif 'HYBRID' in engine_str:
                hp_estimate = int(displacement * 50)  # Hybrid engines (often lower base HP)
            elif 'DIESEL' in engine_str:
                hp_estimate = int(displacement * 55)  # Diesel engines
            else:
                hp_estimate = int(displacement * 50)  # 4-cylinder or other

            return max(hp_estimate, 100)  # Minimum 100 HP

        # Default fallback based on cylinders if available
        return None

    def parse_cylinders(self, engine_str, cylinders_col):
        """Extract number of cylinders from engine description or cylinders column"""
        if pd.notna(cylinders_col) and cylinders_col != '':
            try:
                return int(float(cylinders_col))
            except:
                pass

        if pd.isna(engine_str) or engine_str == '':
            return None

        engine_str = str(engine_str).upper()

        # Look for cylinder patterns
        cyl_patterns = [
            r'V(\d+)',
            r'I(\d+)', 
            r'(\d+)\s*CYL',
            r'(\d+)\s*CYLINDER',
            r'(\d+)V'
        ]

        for pattern in cyl_patterns:
            match = re.search(pattern, engine_str)
            if match:
                return int(match.group(1))

        # If electric motor, return None
        if 'ELECTRIC' in engine_str or 'MOTOR' in engine_str:
            return None

        return 4  # Default to 4 cylinders

    def standardize_fuel_type(self, fuel):
        """Standardize fuel types to match data.csv format"""
        if pd.isna(fuel):
            return 'regular unleaded'

        fuel = str(fuel).lower()

        if 'gasoline' in fuel or 'gas' in fuel:
            if 'premium' in fuel:
                return 'premium unleaded (required)'
            else:
                return 'regular unleaded'
        elif 'diesel' in fuel:
            return 'diesel'
        elif 'electric' in fuel:
            return 'electric'
        elif 'hybrid' in fuel:
            return 'gas/electric hybrid'
        elif 'flex' in fuel or 'e85' in fuel:
            return 'flex-fuel (unleaded/E85)'
        else:
            return 'regular unleaded'

    def standardize_transmission(self, transmission):
        """Standardize transmission types"""
        if pd.isna(transmission):
            return 'AUTOMATIC'

        transmission = str(transmission).upper()

        if 'MANUAL' in transmission:
            return 'MANUAL'
        elif 'CVT' in transmission:
            return 'AUTOMATIC'  # CVT is a type of automatic
        elif 'AUTOMATIC' in transmission:
            return 'AUTOMATIC'
        else:
            return 'AUTOMATIC'  # Default to automatic

    def standardize_drivetrain(self, drivetrain):
        """Standardize drivetrain to match data.csv format"""
        if pd.isna(drivetrain):
            return 'front wheel drive'

        drivetrain = str(drivetrain).lower()

        if 'four' in drivetrain or '4wd' in drivetrain or '4x4' in drivetrain:
            return 'four wheel drive'
        elif 'all' in drivetrain or 'awd' in drivetrain:
            return 'all wheel drive'
        elif 'rear' in drivetrain or 'rwd' in drivetrain:
            return 'rear wheel drive'
        elif 'front' in drivetrain or 'fwd' in drivetrain:
            return 'front wheel drive'
        else:
            return 'front wheel drive'  # Default

    def determine_vehicle_size(self, body, price=None):
        """Determine vehicle size based on body type and price"""
        if pd.isna(body):
            return 'Midsize'

        body = str(body).lower()

        if 'truck' in body:
            return 'Large'
        elif 'suv' in body:
            if price and price > 60000:
                return 'Large'
            else:
                return 'Midsize'
        elif 'van' in body or 'minivan' in body:
            return 'Large'
        elif 'sedan' in body:
            if price and price > 50000:
                return 'Large'
            elif price and price < 25000:
                return 'Compact'
            else:
                return 'Midsize'
        elif 'hatchback' in body or 'convertible' in body:
            return 'Compact'
        else:
            return 'Midsize'  # Default

    def determine_market_category(self, body, price=None, fuel=None, make=None):
        """Determine market category based on various factors"""
        categories = []

        if pd.notna(body):
            body = str(body).lower()
            if 'hatchback' in body:
                categories.append('Hatchback')
            elif 'convertible' in body:
                categories.append('Convertible')

        if pd.notna(fuel):
            fuel = str(fuel).lower()
            if 'hybrid' in fuel:
                categories.append('Hybrid')
            elif 'electric' in fuel:
                categories.append('Electric')
            elif 'diesel' in fuel:
                categories.append('Diesel')
            elif 'flex' in fuel or 'e85' in fuel:
                categories.append('Flex Fuel')

        if pd.notna(make):
            make = str(make).lower()
            luxury_brands = ['bmw', 'mercedes', 'audi', 'lexus', 'acura', 'infiniti', 'cadillac', 'lincoln']
            if any(brand in make for brand in luxury_brands):
                categories.append('Luxury')

        if price:
            if price > 75000:
                categories.append('Luxury')
                if price > 150000:
                    categories.append('Exotic')
            elif price > 50000:
                categories.append('Performance')

        return ','.join(categories) if categories else 'N/A'

    def standardize_vehicle_style(self, body):
        """Standardize vehicle style"""
        if pd.isna(body):
            return 'Sedan'

        body = str(body).lower()

        if 'suv' in body:
            return '4dr SUV'
        elif 'truck' in body:
            if 'crew' in body:
                return 'Crew Cab Pickup'
            else:
                return 'Regular Cab Pickup'
        elif 'sedan' in body:
            return 'Sedan'
        elif 'hatchback' in body:
            return '4dr Hatchback'
        elif 'convertible' in body:
            return 'Convertible'
        elif 'van' in body:
            return 'Passenger Minivan'
        elif 'coupe' in body:
            return 'Coupe'
        else:
            return 'Sedan'  # Default

    def process_vehicles_dataset(self):
        """Process vehicles_dataset.csv to match data.csv structure"""
        if not os.path.exists(self.vehicles_csv_path):
            print(f"Vehicles dataset not found at {self.vehicles_csv_path}")
            return pd.DataFrame()

        vehicles_df = pd.read_csv(self.vehicles_csv_path)
        print(f"Loading vehicles dataset: {vehicles_df.shape}")

        # Create new dataframe with data.csv structure
        processed_df = pd.DataFrame()

        # Direct mappings
        processed_df['Make'] = vehicles_df['make']
        processed_df['Model'] = vehicles_df['model'] 
        processed_df['Year'] = pd.to_numeric(vehicles_df['year'], errors='coerce')
        processed_df['MSRP'] = pd.to_numeric(vehicles_df['price'], errors='coerce')
        processed_df['Number of Doors'] = pd.to_numeric(vehicles_df['doors'], errors='coerce')

        # Parse and standardize complex fields
        processed_df['Engine HP'] = vehicles_df.apply(
            lambda row: self.parse_engine_hp(row['engine']), axis=1
        )

        processed_df['Engine Cylinders'] = vehicles_df.apply(
            lambda row: self.parse_cylinders(row['engine'], row['cylinders']), axis=1
        )

        processed_df['Engine Fuel Type'] = vehicles_df['fuel'].apply(self.standardize_fuel_type)
        processed_df['Transmission Type'] = vehicles_df['transmission'].apply(self.standardize_transmission)
        processed_df['Driven_Wheels'] = vehicles_df['drivetrain'].apply(self.standardize_drivetrain)

        # Handle MPG - the mileage column seems to be a single value, we'll use it for both highway and city
        processed_df['highway MPG'] = pd.to_numeric(vehicles_df['mileage'], errors='coerce')
        processed_df['city mpg'] = processed_df['highway MPG'] * 0.85  # Estimate city MPG as 85% of highway

        # Determine derived fields
        processed_df['Vehicle Size'] = vehicles_df.apply(
            lambda row: self.determine_vehicle_size(row['body'], row['price']), axis=1
        )

        processed_df['Market Category'] = vehicles_df.apply(
            lambda row: self.determine_market_category(row['body'], row['price'], row['fuel'], row['make']), axis=1
        )

        processed_df['Vehicle Style'] = vehicles_df['body'].apply(self.standardize_vehicle_style)

        # Set default popularity (since not available in vehicles dataset)
        processed_df['Popularity'] = 1000  # Default popularity score

        print(f"Processed vehicles dataset: {processed_df.shape}")
        return processed_df

    def load_and_preprocess_data(self):
        """Load and preprocess both datasets, combining them"""
        combined_data = []

        # Load original data.csv if it exists
        if os.path.exists(self.data_csv_path):
            print(f"Loading original data from {self.data_csv_path}...")
            original_df = pd.read_csv(self.data_csv_path)
            print(f"Original dataset shape: {original_df.shape}")
            combined_data.append(original_df)
        else:
            print(f"Original data file not found at {self.data_csv_path}")

        # Process vehicles dataset
        vehicles_df = self.process_vehicles_dataset()
        if not vehicles_df.empty:
            combined_data.append(vehicles_df)

        if not combined_data:
            print("No data files found!")
            return False

        # Combine datasets
        self.df_original = pd.concat(combined_data, ignore_index=True, sort=False)
        print(f"Combined dataset shape: {self.df_original.shape}")

        # Fill missing values in combined dataset
        for col in ['Engine HP', 'Engine Cylinders', 'highway MPG', 'city mpg', 'MSRP', 'Number of Doors']:
            if col in self.df_original.columns:
                self.df_original[col] = pd.to_numeric(self.df_original[col], errors='coerce')

        # Fill missing HP values with estimates
        missing_hp_mask = self.df_original['Engine HP'].isna()
        if missing_hp_mask.any():
            print(f"Estimating HP for {missing_hp_mask.sum()} vehicles...")
            # Estimate based on cylinders and year
            for idx in self.df_original[missing_hp_mask].index:
                cylinders = self.df_original.loc[idx, 'Engine Cylinders']
                year = self.df_original.loc[idx, 'Year']

                if pd.notna(cylinders):
                    base_hp = cylinders * 30  # Base HP per cylinder
                    if pd.notna(year) and year > 2010:
                        base_hp *= 1.2  # Modern engines are more powerful
                    self.df_original.loc[idx, 'Engine HP'] = base_hp
                else:
                    self.df_original.loc[idx, 'Engine HP'] = 200  # Default HP

        # Ensure no zero HP values to avoid division by zero
        self.df_original.loc[self.df_original['Engine HP'] <= 0, 'Engine HP'] = 100

        # Ensure no zero MSRP values to avoid division by zero  
        self.df_original.loc[self.df_original['MSRP'] <= 0, 'MSRP'] = 20000

        # Create derived features with safe division
        self.df_original['Age'] = 2025 - self.df_original['Year']

        # Safe division for Price_per_HP
        self.df_original['Price_per_HP'] = np.where(
            self.df_original['Engine HP'] > 0,
            self.df_original['MSRP'] / self.df_original['Engine HP'],
            self.df_original['MSRP'] / 100  # Default to 100 HP if HP is 0
        )

        # Safe calculation for Efficiency
        avg_mpg = (self.df_original['highway MPG'] + self.df_original['city mpg']) / 2
        self.df_original['Efficiency'] = np.where(
            self.df_original['Engine HP'] > 0,
            (avg_mpg / self.df_original['Engine HP']) * 100,
            avg_mpg  # Default if HP is 0
        )

        # Create the recommendation dataframe with selected features
        all_features = self.numerical_features + self.categorical_features
        available_features = [f for f in all_features if f in self.df_original.columns]
        self.df_rec = self.df_original[available_features + ['Make', 'Model']].copy()

        print(f"Selected {len(available_features)} features for recommendation engine")

        # Handle missing values for numerical features
        for col in self.numerical_features:
            if col in self.df_rec.columns:
                self.df_rec[col] = pd.to_numeric(self.df_rec[col], errors='coerce')
                median_val = self.df_rec[col].median()
                self.df_rec[col].fillna(median_val, inplace=True)

        # Handle missing values for categorical features
        for col in self.categorical_features:
            if col in self.df_rec.columns:
                self.df_rec[col].fillna('Unknown', inplace=True)

        # Special handling for Market Category (as in notebook)
        if 'Market Category' in self.df_rec.columns:
            self.df_rec['Market Category'] = (self.df_rec['Market Category']
                                            .replace(['', ' ', 'NaN', 'N/A', None, np.nan], 'Unknown')
                                            .fillna('Unknown')
                                            .astype(str))

        # Calculate price_to_efficiency with safe division
        self.df_rec['price_to_efficiency'] = np.where(
            self.df_rec['MSRP'] > 0,
            (self.df_rec['Efficiency'] / self.df_rec['MSRP']) * 10000,
            self.df_rec['Efficiency'] * 0.1  # Default if MSRP is 0
        )

        # Handle any remaining infinity or very large values
        for col in self.df_rec.columns:
            if self.df_rec[col].dtype in ['float64', 'float32']:
                # Replace infinity values with NaN, then fill with median
                self.df_rec[col] = self.df_rec[col].replace([np.inf, -np.inf], np.nan)

                # Replace extremely large values (> 1e10) with NaN
                self.df_rec.loc[np.abs(self.df_rec[col]) > 1e10, col] = np.nan

                # Fill NaN with median
                median_val = self.df_rec[col].median()
                if pd.notna(median_val):
                    self.df_rec[col].fillna(median_val, inplace=True)
                else:
                    self.df_rec[col].fillna(0, inplace=True)

        # Add to numerical features
        if 'price_to_efficiency' not in self.numerical_features:
            self.numerical_features.append('price_to_efficiency')

        # Create preprocessing pipelines
        numerical_pipeline = Pipeline(steps=[
            ('impute', SimpleImputer(strategy='median')),
        ])

        categorical_pipeline = Pipeline(steps=[
            ('impute', SimpleImputer(strategy='most_frequent')),
            ('one-hot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])

        # Filter features to only those available in the dataset
        available_numerical = [f for f in self.numerical_features if f in self.df_rec.columns]
        available_categorical = [f for f in self.categorical_features if f in self.df_rec.columns]

        self.preprocessor = ColumnTransformer(transformers=[
            ('num', numerical_pipeline, available_numerical),
            ('cat', categorical_pipeline, available_categorical)
        ], remainder='drop')

        # Create feature matrix
        self.feature_matrix = self.preprocessor.fit_transform(self.df_rec)

        print(f"Data preprocessing completed successfully!")
        print(f"Final dataset shape: {self.df_rec.shape}")
        print(f"Feature matrix shape: {self.feature_matrix.shape}")
        print(f"Numerical features used: {available_numerical}")
        print(f"Categorical features used: {available_categorical}")

        return True

    def filter_by_preferences(self, user_preferences):
        """Filter dataframe based on user preferences (exact notebook implementation)"""
        numerical_features = [
            'Year', 'Engine HP', 'Engine Cylinders', 'highway MPG', 'city mpg',
            'Efficiency', 'Age', 'MSRP', 'Number of Doors', 'Price_per_HP', 'price_to_efficiency'
        ]
        categorical_features = [
            'Market Category', 'Vehicle Size', 'Vehicle Style', 'Transmission Type',
            'Driven_Wheels', 'Engine Fuel Type'
        ]

        filtered_df = self.df_rec.copy()

        for key, value in user_preferences.items():
            if value is not None and key in filtered_df.columns:
                #For MSRP, filter where feature <= value
                if key == 'MSRP':
                    filtered_df = filtered_df[filtered_df[key] <= value]
                elif key in numerical_features:
                    # For numerical features, filter where feature >= value
                    filtered_df = filtered_df[filtered_df[key] >= value]
                elif key in categorical_features:
                    # For categorical features, filter where feature matches value (case-insensitive)
                    if isinstance(value, str):
                        filtered_df = filtered_df[filtered_df[key].str.lower().str.contains(value.lower(), na=False)]

        return filtered_df

    def get_recommendations_by_preference(self, user_preferences, preprocessor=None, feature_matrix=None, df=None, top_n=10):
        """
        Get car recommendations based on user preferences
        This is the exact function from the notebook with parameters for compatibility
        """
        # Use instance variables if parameters not provided
        if preprocessor is None:
            preprocessor = self.preprocessor
        if feature_matrix is None:
            feature_matrix = self.feature_matrix
        if df is None:
            df = self.df_rec

        if preprocessor is None or feature_matrix is None:
            print("Please load and preprocess data first!")
            return pd.DataFrame()

        # Filter the dataframe based on user preferences
        df_filtered = self.filter_by_preferences(user_preferences)

        if df_filtered.empty:
            print("No cars match your preferences. Try relaxing some criteria.")
            return pd.DataFrame()

        # Recreate the feature matrix for filtered data
        filtered_feature_matrix = preprocessor.transform(df_filtered)

        # Create target vector aligned with filtered df
        target = df_filtered['price_to_efficiency'].fillna(0).values

        # Train RandomForest on filtered data
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(filtered_feature_matrix, target)

        # Prepare full user preference vector with expected columns
        expected_columns = df.columns.tolist()
        user_pref_df = pd.DataFrame([{col: None for col in expected_columns}])

        for k, v in user_preferences.items():
            if k in expected_columns:
                user_pref_df.at[0, k] = v

        # Fill missing values in user preferences
        numerical_features = [
            'Year', 'Engine HP', 'Engine Cylinders', 'highway MPG', 'city mpg',
            'Efficiency', 'Age', 'MSRP', 'Number of Doors', 'Price_per_HP', 'price_to_efficiency'
        ]
        categorical_features = [
            'Market Category', 'Vehicle Size', 'Vehicle Style', 'Transmission Type',
            'Driven_Wheels', 'Engine Fuel Type'
        ]

        for col in numerical_features:
            if col in user_pref_df.columns:
                user_pref_df[col] = pd.to_numeric(user_pref_df[col], errors='coerce')
                if col in df.columns:
                    user_pref_df[col].fillna(df[col].median(), inplace=True)

        for col in categorical_features:
            if col in user_pref_df.columns:
                user_pref_df[col].fillna('Unknown', inplace=True)

        # Transform user preferences to feature vector
        user_feature_vector = preprocessor.transform(user_pref_df)

        # Predict similarity scores for filtered vehicles and for user preferences
        vehicle_scores = rf.predict(filtered_feature_matrix)
        user_score = rf.predict(user_feature_vector)[0]

        # Calculate similarity as negative absolute difference from user score (higher is better)
        similarity_scores = -np.abs(vehicle_scores - user_score)

        df_filtered = df_filtered.copy()
        df_filtered['similarity_score'] = similarity_scores

        # Return top_n closest matches
        columns = ['Make', 'Model', 'Year', 'Market Category', 'Transmission Type', 'Vehicle Size',
                  'Engine HP', 'Engine Fuel Type', 'MSRP', 'Efficiency', 'Price_per_HP',
                  'price_to_efficiency', 'similarity_score']

        # Only include columns that exist in the dataframe
        available_columns = [col for col in columns if col in df_filtered.columns]
        result_df = df_filtered.sort_values(by='similarity_score', ascending=False)[available_columns].head(top_n)

        return result_df

    def get_unique_values(self, column_name):
        """Get unique values for a specific column"""
        if self.df_rec is not None and column_name in self.df_rec.columns:
            return sorted([str(x) for x in self.df_rec[column_name].unique() if pd.notna(x)])
        return []

    def create_interactive_interface(self):
        """Create an interactive interface for car recommendations"""
        if not WIDGETS_AVAILABLE:
            print("Interactive interface requires ipywidgets. Install with: pip install ipywidgets")
            return

        if self.df_rec is None:
            print("Please load and preprocess data first!")
            return

        print("Creating interactive interface...")
        print("Note: This requires Jupyter notebook with ipywidgets installed")

        # Create widgets for user input
        widgets_dict = {}

        # Numerical features widgets
        for feature in ['Year', 'Engine HP', 'Engine Cylinders', 'highway MPG', 'city mpg', 'MSRP', 'Number of Doors']:
            if feature in self.df_rec.columns:
                min_val = int(self.df_rec[feature].min())
                max_val = int(self.df_rec[feature].max())
                widgets_dict[feature] = widgets.IntSlider(
                    value=min_val,
                    min=min_val,
                    max=max_val,
                    description=f'{feature}:',
                    style={'description_width': 'initial'},
                    continuous_update=False
                )

        # Categorical features widgets
        for feature in ['Market Category', 'Vehicle Size', 'Vehicle Style', 'Transmission Type', 'Driven_Wheels', 'Engine Fuel Type']:
            if feature in self.df_rec.columns:
                unique_vals = ['Any'] + self.get_unique_values(feature)
                widgets_dict[feature] = widgets.Dropdown(
                    options=unique_vals,
                    value='Any',
                    description=f'{feature}:',
                    style={'description_width': 'initial'}
                )

        # Number of recommendations widget
        widgets_dict['top_n'] = widgets.IntSlider(
            value=10,
            min=5,
            max=20,
            description='Number of recommendations:',
            style={'description_width': 'initial'}
        )

        def show_recommendations(**kwargs):
            """Function to show recommendations based on widget values"""
            clear_output(wait=True)
            # Build user preferences from widget values
            user_prefs = {}
            for key, value in kwargs.items():
                if key != 'top_n' and value != 'Any':
                    user_prefs[key] = value

            print("Selected preferences:")
            for key, value in user_prefs.items():
                print(f" {key}: {value}")

            print("\n" + "="*50 + "\n")

            # Get recommendations
            recommendations = self.get_recommendations_by_preference(user_prefs, top_n=kwargs['top_n'])

            if not recommendations.empty:
                print(f"Top {kwargs['top_n']} Recommended Cars:")
                print("="*50)
                display(recommendations)
            else:
                print("No recommendations found. Try adjusting your preferences.")

        # Create interactive interface
        interactive_plot = interactive(show_recommendations, **widgets_dict)
        display(interactive_plot)

    def get_feature_info(self):
        """Get information about available features"""
        if self.df_rec is None:
            print("Please load and preprocess data first!")
            return

        print("Available Features:")
        print("="*50)
        print("\nNumerical Features:")
        for feature in self.numerical_features:
            if feature in self.df_rec.columns:
                min_val = self.df_rec[feature].min()
                max_val = self.df_rec[feature].max()
                mean_val = self.df_rec[feature].mean()
                print(f" {feature}: {min_val:.2f} - {max_val:.2f} (avg: {mean_val:.2f})")

        print("\nCategorical Features:")
        for feature in self.categorical_features:
            if feature in self.df_rec.columns:
                unique_vals = self.df_rec[feature].unique()
                print(f" {feature}: {len(unique_vals)} unique values")
                print(f" Options: {', '.join(map(str, unique_vals[:5]))}{'...' if len(unique_vals) > 5 else ''}")

    def display_sample_data(self, n=5):
        """Display sample data from the dataset"""
        if self.df_rec is None:
            print("Please load and preprocess data first!")
            return

        print(f"Sample data ({n} random cars):")
        print("="*50)
        sample = self.df_rec.sample(n=min(n, len(self.df_rec)))
        for idx, car in sample.iterrows():
            print(f"\n{car['Make']} {car['Model']} ({car.get('Year', 'N/A')})")
            print(f" Price: ${car.get('MSRP', 0):,}")
            print(f" Engine: {car.get('Engine HP', 'N/A')} HP, {car.get('Engine Cylinders', 'N/A')} cylinders")
            print(f" MPG: {car.get('highway MPG', 'N/A')} highway, {car.get('city mpg', 'N/A')} city")
            print(f" Category: {car.get('Market Category', 'N/A')}")
            print(f" Size: {car.get('Vehicle Size', 'N/A')}")

# Convenience function to create recommendation system
def create_recommendation_system(data_csv_path='data/data.csv', vehicles_csv_path='data/vehicles_dataset.csv'):
    """
    Create and initialize the car recommendation system

    Parameters:
    data_csv_path (str): Path to the original data.csv file
    vehicles_csv_path (str): Path to the vehicles_dataset.csv file

    Returns:
    CarRecommendationEngine: Initialized recommendation engine
    """
    engine = CarRecommendationEngine(data_csv_path, vehicles_csv_path)
    success = engine.load_and_preprocess_data()
    if success:
        return engine
    else:
        print("Failed to initialize recommendation system.")
        return None

# Example usage and demonstration functions
def run_example_recommendations(data_csv_path='data.csv', vehicles_csv_path='vehicles_dataset.csv'):
    """Run example recommendations with different preference sets"""
    print("Car Recommendation Engine - Example Usage")
    print("="*60)

    # Initialize the system
    rec_engine = create_recommendation_system(data_csv_path, vehicles_csv_path)

    if rec_engine is None:
        return

    # Show dataset information
    print("\n1. Dataset Information:")
    rec_engine.get_feature_info()

    # Show sample data
    print("\n2. Sample Cars in Dataset:")
    rec_engine.display_sample_data(3)

    # Example 1: Budget-conscious buyer
    print("\n3. Example 1 - Budget-Conscious Buyer:")
    print("-" * 40)
    budget_prefs = {
        'MSRP': 25000,
        'highway MPG': 30,
        'Vehicle Size': 'Compact',
        'Transmission Type': 'AUTOMATIC'
    }

    print("Preferences:", budget_prefs)
    budget_recs = rec_engine.get_recommendations_by_preference(budget_prefs, top_n=5)

    if not budget_recs.empty:
        print("\nTop 5 Budget-Friendly Recommendations:")
        for idx, car in budget_recs.iterrows():
            mpg_info = f"{car.get('highway MPG', 'N/A')} hwy" if 'highway MPG' in car else 'N/A'
            print(f" • {car['Make']} {car['Model']} ({car.get('Year', 'N/A')}) - ${car.get('MSRP', 0):,} - {mpg_info} MPG")

    # Example 2: Performance enthusiast
    print("\n4. Example 2 - Performance Enthusiast:")
    print("-" * 40)
    performance_prefs = {
        'Engine HP': 250,
        'Market Category': 'Performance',
        'Transmission Type': 'MANUAL'
    }

    print("Preferences:", performance_prefs)
    perf_recs = rec_engine.get_recommendations_by_preference(performance_prefs, top_n=5)

    if not perf_recs.empty:
        print("\nTop 5 Performance Recommendations:")
        for idx, car in perf_recs.iterrows():
            hp_info = f"{car.get('Engine HP', 'N/A')} HP" if 'Engine HP' in car else 'N/A'
            print(f" • {car['Make']} {car['Model']} ({car.get('Year', 'N/A')}) - {hp_info} - ${car.get('MSRP', 0):,}")

    # Example 3: Luxury buyer
    print("\n5. Example 3 - Luxury Buyer:")
    print("-" * 40)
    luxury_prefs = {
        'Market Category': 'Luxury',
        'MSRP': 35000,
        'Vehicle Size': 'Midsize'
    }

    print("Preferences:", luxury_prefs)
    luxury_recs = rec_engine.get_recommendations_by_preference(luxury_prefs, top_n=5)

    if not luxury_recs.empty:
        print("\nTop 5 Luxury Recommendations:")
        for idx, car in luxury_recs.iterrows():
            print(f" • {car['Make']} {car['Model']} ({car.get('Year', 'N/A')}) - ${car.get('MSRP', 0):,}")

    print("\n" + "="*60)
    print("Examples completed!")
    print("\nTo use this system in your code:")
    print("1. rec_engine = create_recommendation_system('data.csv', 'vehicles_dataset.csv')")
    print("2. preferences = {'Vehicle Size': 'Compact', 'Engine HP': 200}")
    print("3. recommendations = rec_engine.get_recommendations_by_preference(preferences)")

    return rec_engine

# Main execution
if __name__ == "__main__":
    run_example_recommendations()
