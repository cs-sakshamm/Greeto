from app.vision.embedding_model import EmbeddingModel
from app.vision.face_database import FaceDatabase


def main():

    model = EmbeddingModel()

    database = FaceDatabase(model)

    database.load()

    print()

    print(database.list_people())

    print(database.person_count())

    print(database.embedding_count())


if __name__ == "__main__":
    main()