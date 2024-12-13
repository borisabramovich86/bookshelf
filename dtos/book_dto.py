from pydantic import BaseModel

class DetectedBook(BaseModel):
    title: str
    author: str

class DetectedBooks(BaseModel):
    books: list[DetectedBook]


class FavouriteBook(BaseModel):
    title: str
    author: str
    rating: int
    isbn: str
    publisher: str

class FavouriteBooks(BaseModel):
    books: list[FavouriteBook]