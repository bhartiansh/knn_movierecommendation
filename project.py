import streamlit as st
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from fuzzywuzzy import process

# Load datasets
movies = pd.read_csv("movies.csv")
ratings = pd.read_csv("ratings.csv")

# Data Preprocessing
movies = movies[['movieId', 'title', 'genres']]
ratings = ratings[['userId', 'movieId', 'rating']]

# Pivot ratings table to have movies as rows and users as columns
movies_users = ratings.pivot(index='movieId', columns='userId', values='rating').fillna(0)

# One-hot encoding of genres (split by '|')
movies['genre_list'] = movies['genres'].str.split('|')
genre_matrix = movies['genre_list'].str.join('|').str.get_dummies()

# Set index of genre_matrix to movieId to match the movies_users matrix
genre_matrix = genre_matrix.set_index(movies['movieId'])

# Combine ratings and genre data
combined_matrix = pd.concat([movies_users, genre_matrix], axis=1).fillna(0)

# Convert combined matrix to sparse matrix
mat_combined = csr_matrix(combined_matrix.values)

# Train Nearest Neighbors model
model_combined = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20)
model_combined.fit(mat_combined)

# Fit the model with the user matrix (ratings only)
mat_movies = csr_matrix(movies_users.values)
model_user = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20)
model_user.fit(mat_movies)

# Fit the model with the genre matrix (content-based filtering)
mat_genres = csr_matrix(genre_matrix.values)
model_genre = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20)
model_genre.fit(mat_genres)

# Recommender function for user-based filtering
def recommender_user(movie_name, data, n=10):
    idx = process.extractOne(movie_name, movies['title'])[2]
    st.write(f"**Movie Selected:** {movies['title'][idx]}")
    
    # Find nearest neighbors using the user matrix (ratings only)
    distance, indices = model_user.kneighbors(data[idx], n_neighbors=n)
    
    recommendations = []
    for i in indices[0]:
        if i != idx:
            recommendations.append(movies['title'].iloc[i])
    
    return recommendations

# Recommender function for genre-based filtering
def recommender_genre(movie_name, data, n=10):
    idx = process.extractOne(movie_name, movies['title'])[2]
    st.write(f"**Movie Selected:** {movies['title'][idx]}")
    
    # Find nearest neighbors using the genre matrix (content-based)
    distance, indices = model_genre.kneighbors(data[idx], n_neighbors=n)
    
    recommendations = []
    for i in indices[0]:
        if i != idx:
            recommendations.append(movies['title'].iloc[i])
    
    return recommendations

# Recommender Function using both genre and user-based filtering
def recommender_both(movie_name, data, n=10):
    idx = process.extractOne(movie_name, movies['title'])[2]
    st.write(f"**Movie Selected:** {movies['title'][idx]}")
    
    # Find nearest neighbors using the combined matrix (hybrid approach)
    distance, indices = model_combined.kneighbors(data[idx], n_neighbors=n)

    recommendations = []
    for i in indices[0]:
        if i != idx:
            recommendations.append(movies['title'].iloc[i])
    
    return recommendations

# Front-End Interface using Streamlit
def main():
    # Title of the web app
    st.title("Movie Recommendation System")
    
    # Introduction text
    st.write("""
        **Movie Recommender System:**  
        This system provides recommendations based on **user ratings**, **movie genres**, or a **hybrid** of both. 
        Choose your preferred method below.
    """)
    
    # Movie selection
    movie_list = movies['title'].values
    selected_movie = st.selectbox("Choose a movie:", movie_list)

    # Selection for recommendation method
    option = st.selectbox(
        "Choose a recommendation method:",
        ("User-Based Filtering", "Genre-Based Filtering", "(User + Genre)")
    )
    
    # Recommendation button
    if st.button("Recommend"):
        if option == "User-Based Filtering":
            st.write("Using **User-Based Filtering** for recommendations:")
            recommendations = recommender_user(selected_movie, mat_movies, 10)
        elif option == "Genre-Based Filtering":
            st.write("Using **Genre-Based Filtering** for recommendations:")
            recommendations = recommender_genre(selected_movie, mat_genres, 10)
        else:
            st.write("Using **(User + Genre)** for recommendations:")
            recommendations = recommender_both(selected_movie, mat_combined, 10)
        
        # Display the recommendations
        for i, movie in enumerate(recommendations, 1):
            st.write(f"{i}. {movie}")

# Run the app
if __name__ == '__main__':
    main()