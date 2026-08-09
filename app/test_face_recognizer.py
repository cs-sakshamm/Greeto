from app.vision.embedding_model import EmbeddingModel
from app.vision.face_database import FaceDatabase
from app.vision.face_recognizer import FaceRecognizer
import cv2

embedding_model = EmbeddingModel()

database = FaceDatabase(embedding_model)
database.load()

recognizer = FaceRecognizer(
    embedding_model,
    database,
)

image = cv2.imread("models/faces/Saksham/1.jpg")

result = recognizer.recognize(image)

print(result)