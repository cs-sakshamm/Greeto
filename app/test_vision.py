from app.core.container import ServiceContainer
from app.vision.vision_service import VisionService

container = ServiceContainer()

vision = VisionService(container)

if vision.start():

    print("Vision started successfully.")

    vision.stop()

else:

    print("Failed to start vision.")