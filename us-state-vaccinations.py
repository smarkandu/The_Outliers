import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns

from datetime import datetime

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

df = pd.read_csv(os.path.join('data', 'us_state_vaccinations.csv'))

print(df.head(50))
print(df.sample(5))
print(df.describe())
print(df.info())

sns.barplot(data=df[df['location'] != 'United States'], x='location', y='total_vaccinations')
plt.xticks(rotation=90)
plt.show()

df['date'] = df['date'].map(lambda value: datetime.strptime(value, '%Y-%m-%d'))

sns.lineplot(data=df, x='date', y='daily_vaccinations')
plt.xticks(rotation=90)
plt.show()


encoder = ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['location'])
imputer = ('imputer', SimpleImputer(), slice(2, 16))

transformer = ColumnTransformer(transformers=[encoder, imputer], remainder='passthrough')
transformer.set_output(transform='pandas')

df = pd.DataFrame(transformer.fit_transform(df))

print(df.info())
print(df.corr())
print(np.fill_diagonal(df.corr().values, 0))
