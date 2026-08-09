from PySide6.QtGui import QColor

WINDOW_BG = "#0B1220"

CARD_BG = "#151E2F"

CARD_BG2 = "#101827"

TEXT = "#FFFFFF"

ACCENT = "#3B82F6"

STYLE = f"""

QMainWindow{{
    background:{WINDOW_BG};
}}

QWidget{{
    background:{WINDOW_BG};
    color:{TEXT};
    font-family:Segoe UI;
    font-size:14px;
}}

QFrame{{
    background:{CARD_BG};
    border:none;
    border-radius:22px;
}}

QLabel{{
    background:transparent;
    color:white;
}}

QTextEdit{{
    background:{CARD_BG};
    border:none;
    border-radius:18px;
    padding:15px;
    color:white;
}}

QScrollBar:vertical{{
    width:0px;
}}

"""