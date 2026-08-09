from app.core.container import ServiceContainer
from app.speech.speech_service import SpeechService


def main():

    container = ServiceContainer()

    speech = SpeechService(
        container
    )

    speech.start()

    try:

        while True:
            pass

    except KeyboardInterrupt:

        speech.stop()


if __name__ == "__main__":
    main()