
#!/usr/bin/env python3
"""
Simple Car Recommendation Script

This script demonstrates direct usage of the recommendation system
with the exact same data paths and function calls as in the notebook.
"""

import os
import sys
import pandas as pd
import numpy as np
from car_recommendation_system import CarRecommendationEngine

def main():
    print("Car Recommendation System - Direct Usage")
    print("="*50)

    # Check if data file exists
    data_path = 'data/data.csv'
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at '{data_path}'")
        print("Please ensure the data file is in the correct location.")
        print("\nExpected directory structure:")
        print("├── car_recommendation_system.py")
        print("├── data/")
        print("│   └── data.csv")
        print("└── notebooks/")
        print("    └── rec_engine.ipynb")
        return

    # Initialize the recommendation engine
    print(f"Loading data from {data_path}...")
    rec_engine = CarRecommendationEngine(data_path)

    # Load and preprocess data
    success = rec_engine.load_and_preprocess_data()
    if not success:
        print("Failed to load and preprocess data.")
        return

    print("Recommendation engine initialized successfully!")
    print(f"Dataset contains {len(rec_engine.df_rec)} cars")

    # Display available features
    print("\n" + "="*50)
    print("AVAILABLE FEATURES:")
    print("="*50)
    rec_engine.get_feature_info()

    # Show sample cars
    print("\n" + "="*50)
    print("SAMPLE CARS IN DATASET:")
    print("="*50)
    rec_engine.display_sample_data(3)

    # Example recommendations using the exact notebook function signature
    print("\n" + "="*50)
    print("EXAMPLE RECOMMENDATIONS:")
    print("="*50)

    # Example 1: Exact notebook usage
    print("\n1. Using exact notebook function signature:")
    print("-" * 30)
    user_preferences = {
        'Vehicle Style': '4dr Hatchback',
        'Vehicle Size': 'Compact', 
        'Engine HP': 100,
        'Year': 2015,
        'Transmission Type': 'Automatic',
        'Market Category': 'Hatchback,Hybrid'
    }

    print(f"Preferences: {user_preferences}")

    # Call with exact notebook signature: get_recommendations_by_preference(user_preferences, preprocessor, feature_matrix, df, top_n=10)
    recommendations = rec_engine.get_recommendations_by_preference(
        user_preferences,
        rec_engine.preprocessor,
        rec_engine.feature_matrix, 
        rec_engine.df_rec,
        top_n=10
    )

    if not recommendations.empty:
        print(f"\n✓ Found {len(recommendations)} recommendations:")
        for idx, car in recommendations.head().iterrows():
            price = car.get('MSRP', 0)
            hp = car.get('Engine HP', 'N/A')
            year = car.get('Year', 'N/A')
            score = car.get('similarity_score', 0)
            print(f"  • {car['Make']} {car['Model']} ({year}) - ${price:,} - {hp} HP - Score: {score:.3f}")
    else:
        print("No recommendations found.")

    # Example 2: Budget-conscious buyer
    print("\n2. Budget-conscious buyer:")
    print("-" * 30)
    budget_prefs = {
        'MSRP': 25000,      # Max budget
        'highway MPG': 30,   # Good fuel economy
        'Vehicle Size': 'Compact',
        'Transmission Type': 'AUTOMATIC'
    }
    print(f"Preferences: {budget_prefs}")

    budget_recs = rec_engine.get_recommendations_by_preference(budget_prefs, top_n=5)
    if not budget_recs.empty:
        print("\n✓ Budget-friendly options:")
        for idx, car in budget_recs.iterrows():
            price = car.get('MSRP', 0)
            mpg = car.get('highway MPG', 'N/A')
            print(f"  • {car['Make']} {car['Model']} - ${price:,} - {mpg} hwy MPG")

    # Example 3: Performance enthusiast
    print("\n3. Performance enthusiast:")
    print("-" * 30)
    perf_prefs = {
        'Engine HP': 250,
        'Market Category': 'Performance',
        'Year': 2010  # Minimum year
    }
    print(f"Preferences: {perf_prefs}")

    perf_recs = rec_engine.get_recommendations_by_preference(perf_prefs, top_n=5)
    if not perf_recs.empty:
        print("\n✓ High-performance options:")
        for idx, car in perf_recs.iterrows():
            hp = car.get('Engine HP', 'N/A')
            price = car.get('MSRP', 0)
            print(f"  • {car['Make']} {car['Model']} - {hp} HP - ${price:,}")

    # Interactive section
    print("\n" + "="*50)
    print("INTERACTIVE MODE:")
    print("="*50)
    print("Enter your preferences (or press Enter to use defaults):")

    try:
        # Simple interactive input
        interactive_prefs = {}

        make = input("Preferred Make (e.g., Toyota, Honda) [Any]: ").strip()
        if make:
            interactive_prefs['Make'] = make

        size = input("Vehicle Size (Compact/Midsize/Large) [Any]: ").strip()
        if size:
            interactive_prefs['Vehicle Size'] = size

        max_price = input("Maximum Price [50000]: ").strip()
        if max_price and max_price.isdigit():
            interactive_prefs['MSRP'] = int(max_price)
        elif not max_price:
            interactive_prefs['MSRP'] = 50000

        min_hp = input("Minimum Horsepower [150]: ").strip()
        if min_hp and min_hp.isdigit():
            interactive_prefs['Engine HP'] = int(min_hp)
        elif not min_hp:
            interactive_prefs['Engine HP'] = 150

        if interactive_prefs:
            print(f"\nYour preferences: {interactive_prefs}")

            interactive_recs = rec_engine.get_recommendations_by_preference(interactive_prefs, top_n=5)

            if not interactive_recs.empty:
                print("\n✓ Your personalized recommendations:")
                for idx, car in interactive_recs.iterrows():
                    price = car.get('MSRP', 0)
                    hp = car.get('Engine HP', 'N/A') 
                    year = car.get('Year', 'N/A')
                    score = car.get('similarity_score', 0)
                    print(f"  • {car['Make']} {car['Model']} ({year}) - ${price:,} - {hp} HP - Score: {score:.3f}")

                # Save to CSV
                save = input("\nSave recommendations to CSV? (y/n) [n]: ").strip().lower()
                if save == 'y':
                    filename = 'my_recommendations.csv'
                    interactive_recs.to_csv(filename, index=False)
                    print(f"Recommendations saved to '{filename}'")
            else:
                print("No cars match your preferences. Try adjusting the criteria.")
        else:
            print("No preferences entered. Skipping interactive recommendations.")

    except KeyboardInterrupt:
        print("\n\nExiting...")
    except Exception as e:
        print(f"\nError in interactive mode: {e}")

    print("\n" + "="*50)
    print("Demo completed!")
    print("\nTo use this system in your own code:")
    print("```python")
    print("from car_recommendation_system import CarRecommendationEngine")
    print("")
    print("# Initialize")
    print("engine = CarRecommendationEngine('data/data.csv')")
    print("engine.load_and_preprocess_data()")
    print("")
    print("# Get recommendations")
    print("prefs = {'Vehicle Size': 'Compact', 'MSRP': 30000}")
    print("recs = engine.get_recommendations_by_preference(prefs)")
    print("```")

if __name__ == "__main__":
    main()
