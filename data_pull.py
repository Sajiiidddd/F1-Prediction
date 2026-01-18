#imports
import fastf1
import pandas as pd

#setting cache 
fastf1.Cache.enable_cache('cache')

#loading session data
#bahrain gp 2023 qualy
session = fastf1.get_session(2023, 'Bahrain', 'R')
session.load()

#print result
results = session.results
print(results.columns)

# Select only the drivers who finished the race
finished_drivers = results[results['Status'] == 'Finished']

# Create a new column to calculate places gained or lost
# Note: A negative number is good (gained places), a positive number is bad (lost places)
finished_drivers['PlacesGained'] = finished_drivers['Position'] - finished_drivers['GridPosition']

# Show the result for a few key columns
print(finished_drivers[['FullName', 'GridPosition', 'Position', 'PlacesGained']].sort_values('PlacesGained'))