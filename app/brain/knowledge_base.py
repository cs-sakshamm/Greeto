"""
Knowledge Base

Loads Markdown, TXT and JSON files from app/knowledge and performs
simple relevance-based retrieval for the local LLM.
"""

from __future__ import annotations

import json
import re
from pathlib import Path


class KnowledgeBase:

    def __init__(self) -> None:

        self.documents: list[dict] = []

        self.knowledge_folder = (
            Path(__file__).resolve().parent.parent / "knowledge"
        )

        self.load()

    # ------------------------------------------------------------

    def load(self) -> None:

        self.documents.clear()

        if not self.knowledge_folder.exists():

            print("Knowledge folder not found.")

            return

        # Markdown
        for file in sorted(self.knowledge_folder.glob("*.md")):

            try:

                text = file.read_text(
                    encoding="utf-8",
                    errors="ignore",
                )

                self._load_markdown(
                    file.name,
                    text,
                )

            except Exception as e:

                print(f"Failed loading {file.name}: {e}")

        # TXT
        for file in sorted(self.knowledge_folder.glob("*.txt")):

            try:

                self.documents.append(
                    {
                        "source": file.name,
                        "section": file.stem,
                        "content": file.read_text(
                            encoding="utf-8",
                            errors="ignore",
                        ),
                    }
                )

            except Exception:

                pass

        # JSON
        for file in sorted(self.knowledge_folder.glob("*.json")):

            try:

                data = json.loads(
                    file.read_text(
                        encoding="utf-8",
                        errors="ignore",
                    )
                )

                self.documents.append(
                    {
                        "source": file.name,
                        "section": file.stem,
                        "content": json.dumps(
                            data,
                            indent=2,
                            ensure_ascii=False,
                        ),
                    }
                )

            except Exception:

                pass

        print(f"KnowledgeBase loaded {len(self.documents)} sections.")

    # ------------------------------------------------------------

    def _load_markdown(
        self,
        filename: str,
        text: str,
    ) -> None:

        if not text.strip():
            return

        current_title = "Introduction"

        current_body = []

        for line in text.splitlines():

            heading = re.match(r"^\s*#{1,6}\s+(.*)", line)

            if heading:

                if current_body:

                    self.documents.append(
                        {
                            "source": filename,
                            "section": current_title,
                            "content": "\n".join(current_body).strip(),
                        }
                    )

                current_title = heading.group(1).strip()

                current_body = []

            else:

                current_body.append(line)

        if current_body:

            self.documents.append(
                {
                    "source": filename,
                    "section": current_title,
                    "content": "\n".join(current_body).strip(),
                }
            )

    # ------------------------------------------------------------

    def search(
        self,
        query: str,
    ) -> str:

        if not query.strip():

            return ""

        words = {

            w.lower()

            for w in re.findall(
                r"[a-zA-Z0-9]+",
                query,
            )

            if len(w) > 2

        }

        if not words:

            return ""

        results = []

        for doc in self.documents:

            searchable = (
                f"{doc['section']} {doc['content']}"
            ).lower()

            score = 0

            for word in words:

                if word in searchable:

                    score += 10

                if word.endswith("s"):

                    if word[:-1] in searchable:

                        score += 5

                else:

                    if word + "s" in searchable:

                        score += 5

            if score:

                results.append((score, doc))

        if not results:

            return ""

        results.sort(
            key=lambda x: x[0],
            reverse=True,
        )

        context = []

        for _, doc in results[:3]:

            context.append(
                f"""Source: {doc['source']}
Section: {doc['section']}

{doc['content']}"""
            )

        return "\n\n".join(context)