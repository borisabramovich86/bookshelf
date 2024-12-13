from recommenders.recommender import BookRecommender
from recommenders.ollama_recommender import OllamaBookRecommender

class RecommenderFactory:
    def get_recommender(self, type: str) -> BookRecommender:
        if type == 'ollama':
            return OllamaBookRecommender
        else:
            return ValueError(type)