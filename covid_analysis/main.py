import pandas as pd

us_vaccinations_df = pd.read_csv('./data/us_state_vaccinations.csv')
print(len(us_vaccinations_df))
print(us_vaccinations_df.dtypes)
print(us_vaccinations_df.describe())
print('\n\n')

