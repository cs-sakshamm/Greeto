"""
Visitor Database.

Stores visitor information.
"""

from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime



class VisitorDatabase:
    """
    Persistent visitor storage.
    """


    def __init__(self):

        self.file = Path(
            "data/visitors.json"
        )

        self.visitors = {}

        self.load()



    # ---------------------------------------------

    def load(self):

        self.file.parent.mkdir(
            exist_ok=True
        )


        if self.file.exists():

            self.visitors = json.loads(

                self.file.read_text(
                    encoding="utf-8"
                )

            )



    # ---------------------------------------------

    def save(self):

        self.file.write_text(

            json.dumps(
                self.visitors,
                indent=4
            ),

            encoding="utf-8"

        )



    # ---------------------------------------------

    def register(
        self,
        name: str,
    ):

        now = datetime.now()


        if name not in self.visitors:

            self.visitors[name] = {

                "first_seen":
                    now.isoformat(),

                "visits": 1,

                "last_seen":
                    now.isoformat(),

            }


        else:

            self.visitors[name]["visits"] += 1

            self.visitors[name]["last_seen"] = (
                now.isoformat()
            )



        self.save()



    # ---------------------------------------------

    def get(
        self,
        name: str,
    ):

        return self.visitors.get(
            name
        )