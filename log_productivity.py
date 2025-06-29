import pandas as pd
import numpy as np
import random

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# Function to simulate productivity dataset
def generate_dummy_productivity_data(n_samples=150):
    data = []
    for _ in range(n_samples):
        wake_hour = random.choice(range(5, 11))           # 5 to 10 AM
        sleep_hours = round(random.uniform(4.5, 9.0), 1)   # Sleep between 4.5 to 9 hours
        screen_time = round(random.uniform(2.0, 9.0), 1)   # Screen time in hours
        exercise = random.choice([0, 15, 30, 45, 60])      # Minutes of exercise
        weather = random.choice(['Sunny', 'Cloudy', 'Rainy'])
        meals = random.choice([2, 3, 4])
        
        # Logic to label productivity
        is_productive = int(
            (6.5 <= sleep_hours <= 8.5) and
            (screen_time <= 6.0) and
            (exercise >= 15) and
            (meals >= 3)
        )
        
        data.append([wake_hour, sleep_hours, screen_time, exercise, weather, meals, is_productive])
    
    columns = ['WakeTimeHour', 'SleepHours', 'ScreenTime', 'ExerciseMins', 'Weather', 'Meals', 'Productive']
    return pd.DataFrame(data, columns=columns)

# Generate and save dataset
df = generate_dummy_productivity_data(150)
df.to_csv('personal_productivity_dataset.csv', index=False)
print("✅ Dummy dataset generated and saved as 'personal_productivity_dataset.csv'")
