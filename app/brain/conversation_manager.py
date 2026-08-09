"""
Conversation Manager.

Controls AI receptionist conversation.

Flow:

Speech
 ↓
Conversation Manager
 ↓
Knowledge Base
 ↓
Prompt Manager
 ↓
Local LLM
 ↓
TTS
"""


from __future__ import annotations


import threading
import time


from app.core.container import ServiceContainer


from app.brain.llm import LocalLLM
from app.brain.prompt_manager import PromptManager
from app.brain.parser import ResponseParser
from app.brain.knowledge_base import KnowledgeBase


from app.speech.tts_service import TTSService




class ConversationManager:
    """
    Main AI conversation controller.
    """



    def __init__(
        self,
        container: ServiceContainer,
    ) -> None:


        self.container = container

        self.logger = container.logger

        self.event_bus = container.event_bus



        self.llm = LocalLLM()


        self.prompt_manager = PromptManager()


        self.parser = ResponseParser()


        self.knowledge = KnowledgeBase()


        self.tts = TTSService(
            container
        )



        self.history = []



        self.processing_lock = threading.Lock()


        self.last_request = 0



        self.register_events()



        self.logger.info(
            "Conversation Manager started."
        )



    # -------------------------------------------------

    def register_events(
        self,
    ):

        self.event_bus.subscribe(

            "speech_recognized",

            self.handle_speech

        )



    # -------------------------------------------------

    def handle_speech(
        self,
        text: str,
    ):


        if not text:

            return



        text = text.strip()



        ignored = [

            "[blank_audio]",

            "blank_audio",

            "silence",

            "[silence]",

        ]



        if text.lower() in ignored:

            return



        if len(text) < 3:

            return



        now = time.time()



        if now - self.last_request < 1:

            return



        self.last_request = now



        threading.Thread(

            target=self.process_message,

            args=(text,),

            daemon=True,

        ).start()




    # -------------------------------------------------

    def process_message(
        self,
        text: str,
    ):


        with self.processing_lock:


            try:


                self.logger.info(
                    f"User: {text}"
                )



                self.history.append(

                    {

                        "role": "user",

                        "content": text

                    }

                )



                if len(self.history) > 6:

                    self.history = self.history[-6:]




                context = self.knowledge.search(
                    text
                )
                self.logger.info(f"Retrieved context length: {len(context)}")

                if context:
                  self.logger.info(f"Context preview:\n{context[:500]}")
                else:
                  self.logger.info("Context is EMPTY")


                prompt = self.prompt_manager.build_prompt(

                    user_message=text,

                    history=self.history,

                    context=context,

                )



                self.logger.info(
                    "Generating AI response..."
                )



                response_text = ""



                for token in self.llm.stream_generate(

                    prompt

                ):


                    response_text += token



                self.logger.info(

                    f"LLM Raw: {response_text}"

                )



                response = self.parser.parse(

                    response_text

                )



                if response is None:

                    return



                answer = response.reply.strip()



                self.history.append(

                    {

                        "role": "assistant",

                        "content": answer

                    }

                )



                self.event_bus.publish(

                    "llm_response_ready",

                    response=response

                )



                self.logger.info(

                    f"Speaking: {answer}"

                )



                self.tts.speak(

                    answer

                )



            except Exception as error:


                self.logger.exception(

                    error

                )



    # -------------------------------------------------

    def clear_history(
        self,
    ):

        self.history.clear()