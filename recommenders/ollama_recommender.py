from recommenders.recommender import BookRecommender
import pandas as pd
from ollama import chat
from dtos.book_dto import FavouriteBook, DetectedBook, DetectedBooks
import logging

class OllamaBookRecommender(BookRecommender):
    def __init__(self):
        super().__init__()

    def get_recommendations(self, favorite_books: list[FavouriteBook], new_books: list[DetectedBook]) -> str:
        favorite_books_str = "\n".join([f"Title: {book.title}, Author: {book.author}, Rating: {book.rating}" for book in favorite_books])
        new_books_str = "\n".join([f"Title: {book.title}, Author: {book.author}" for book in new_books])

        prompt = f"""
        I have rated the following books:

        {favorite_books_str}

        Based on my ratings, here is a list of new books I am considering:

        {new_books_str}

        Which of these new books would you recommend to me and why?
        Output only the title and author of the recommended books.
        """

        response = chat(model='llama3.2',
                        format=DetectedBooks.model_json_schema(),
                        messages=[{"role": "user", "content": prompt}])
        recommended_books = DetectedBooks.model_validate_json(response.message.content)
        return recommended_books

    def recommend(self, new_books):
        logging.info('Parsing favourite books')
        csv_file = 'resources/book_data/my_books.csv'
        df = pd.read_csv(csv_file, keep_default_na=False, names=['Title', 'Author', 'ISBN', 'MyRating', 'Publisher'])

        favorite_books = [FavouriteBook(
            title=row['Title'], 
            author=row['Author'],
            isbn=row['ISBN'],
            rating=row['MyRating'],
            publisher=row['Publisher']) for _, row in df.iterrows()]
        
        logging.info('Recommending new books using ollama')
        recommendations = self.get_recommendations(favorite_books, new_books)

        logging.info("\nRecommendations:\n")
        logging.info(recommendations)
        return recommendations