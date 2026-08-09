"""
Vision Service.

Optimized real-time vision pipeline.

Pipeline:

Camera
    ↓
FrameGrabber
    ↓
Vision Processing
    ↓
Detection (periodic)
    ↓
Recognition
    ↓
Tracking
    ↓
Display
    ↓
EventBus
"""

from __future__ import annotations

import time

import cv2
import numpy as np

from app.core.container import ServiceContainer

from app.vision.camera_manager import CameraManager
from app.vision.embedding_model import EmbeddingModel
from app.vision.face_database import FaceDatabase
from app.vision.face_recognizer import FaceRecognizer
from app.vision.frame_grabber import FrameGrabber
from app.vision.models import Face
from app.vision.visitor_tracker import VisitorTracker
from app.vision.visitor_database import VisitorDatabase


class VisionService:
    """
    Optimized vision service.
    """


    def __init__(
        self,
        container: ServiceContainer,
    ) -> None:

        self.container = container

        self.logger = container.logger

        self.event_bus = getattr(
            container,
            "event_bus",
            None,
        )


        self.camera = CameraManager(
            container
        )


        self.grabber = FrameGrabber(
            self.camera
        )


        self.embedding_model = EmbeddingModel()


        self.database = FaceDatabase(
            self.embedding_model
        )


        self.recognizer = FaceRecognizer(
            self.database
        )


        self.tracker = VisitorTracker()
        self.visitor_database = VisitorDatabase()


        self._running = False


        self._latest_frame = None


        # Performance control

        self.frame_counter = 0

        self.detection_interval = 3


        self.cached_faces: list[Face] = []


        self.last_time = time.perf_counter()

        self.fps = 0.0



    # -----------------------------------------------------

    def start(self) -> bool:

        self.logger.info(
            "Starting Vision Service..."
        )


        if not self.camera.open():

            return False


        self.database.load()


        self.grabber.start()


        self._running = True


        self.logger.info(
            "Vision Service started."
        )


        return True



    # -----------------------------------------------------

    def stop(self):

        self._running = False


        self.grabber.stop()


        self.camera.release()


        cv2.destroyAllWindows()


        self.logger.info(
            "Vision stopped."
        )



    # -----------------------------------------------------

    def process(
        self,
    ) -> np.ndarray | None:


        if not self._running:

            return None



        frame = self.grabber.get_frame()


        if frame is None:

            return None



        self.frame_counter += 1



        try:


            #
            # Heavy AI processing only periodically
            #

            if (
                self.frame_counter
                % self.detection_interval
                == 0
            ):


                faces = self.embedding_model.detect(
                    frame
                )


                faces = self.recognizer.recognize_all(
                    faces
                )


                faces = self.tracker.update(
                    faces
                )


                self.cached_faces = faces


                self._publish_events(
                    faces
                )



            else:

                faces = self.cached_faces



            output = frame.copy()


            self._draw_faces(
                output,
                faces,
            )


            self._draw_fps(
                output
            )


            self._latest_frame = output


            return output



        except Exception as error:

            self.logger.exception(error)

            return frame



    # -----------------------------------------------------

    def _draw_faces(
        self,
        frame,
        faces,
    ):


        for face in faces:


            color = (
                (0,255,0)
                if face.recognized
                else
                (0,165,255)
            )


            cv2.rectangle(
                frame,
                (
                    face.x,
                    face.y
                ),
                (
                    face.x + face.width,
                    face.y + face.height
                ),
                color,
                2,
            )


            label = (
                f"{face.name} "
                f"{face.confidence:.2f}"
            )


            cv2.putText(
                frame,
                label,
                (
                    face.x,
                    face.y - 10
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )



    # -----------------------------------------------------

    def _draw_fps(
        self,
        frame,
    ):

        now = time.perf_counter()


        elapsed = now - self.last_time


        if elapsed > 0:

            self.fps = (
                0.9 * self.fps
                +
                0.1 * (1 / elapsed)
            )


        self.last_time = now


        cv2.putText(
            frame,
            f"FPS: {self.fps:.1f}",
            (10,30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0,255,0),
            2,
        )



    # -----------------------------------------------------

    def _publish_events(
        self,
        faces: list[Face],
    ):


        if self.event_bus is None:

            return



        for face in faces:


            if face.recognized:

             self.visitor_database.register(
               face.name
    )


             self.event_bus.publish(
                   "visitor.detected",
                  name=face.name,
                  face=face,
    )
             
            else:

                     self.event_bus.publish(
                        "visitor.unknown",
                            face=face,
    )



    # -----------------------------------------------------

   
        # -----------------------------------------------------

    def show(
        self,
        window_name="Vision",
    ) -> bool:
        """
        Display is handled by the Qt GUI.
        This method is kept only for compatibility.
        """
        return True





        


    



    # -----------------------------------------------------

        # -----------------------------------------------------

    def run(self):
        """
        Starts the vision pipeline.

        Frame updates are handled by the Qt GUI
        (CameraWidget + QTimer), so there is no
        processing loop here.
        """

        if not self.start():
            return

        self.logger.info(
            "Vision service running under Qt GUI."
        )