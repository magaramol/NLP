import pandas as pd

df=pd.read_csv('/home/ams/Documents/python/vscode/nlp/NLP/OMDB_data/movies.csv')


import requests
import pandas as pd


# API key for OMDB API
api_key = 'api_key'

# URL for OMDB API
url = "http://www.omdbapi.com/"

# List to store movie data
movie_data_list = []

# Iterate over rows in DataFrame
for index, row in df.iterrows():
    title = row['movie']
    params = {
        't': title,
        'apikey': api_key
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        try:
            movie_data = response.json()
            movie_info = {
                'Title': movie_data['Title'],
                'Year': movie_data['Year'],
                'Plot': movie_data['Plot']
            }
            movie_data_list.append(movie_info)
        except KeyError as e:
            print(f"Error processing {title}: {e}")
    else:
        print(f"Failed to fetch data for {title}")

# Create a DataFrame from the list of movie data
movie_df = pd.DataFrame(movie_data_list)

# Save the DataFrame to a CSV file
movie_df.to_csv('movie_data.csv', index=False)
