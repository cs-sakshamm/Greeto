"""
Application Theme
PyQt5
"""

# Style used by the previous card-based interface. Kept so any older
# caller that imports STYLE keeps working; the kiosk uses KIOSK_STYLE.
STYLE = """
*{
    font-family: "Segoe UI";
    color:white;
}

QMainWindow{
    background:#0D1117;
}

QWidget{
    background:#0D1117;
}

QFrame#RobotCard{
    background:#0D1117;
    border:none;
}

QLabel#Caption{

    font-size:28px;

    font-weight:500;

    color:white;

    padding:20px;

    background:transparent;
}

QLabel#Robot{

    background:transparent;
}

QLabel{

    background:transparent;
}

QPushButton{

    border:none;

    background:#2563EB;

    border-radius:12px;

    padding:10px;

    font-size:16px;
}

QPushButton:hover{

    background:#3B82F6;
}

QPushButton:pressed{

    background:#1D4ED8;
}
"""


# Kiosk style. Deliberately minimal: the background is painted by
# KioskBackground and the face by RobotWidget, so every QWidget that is
# not one of those only needs a transparent background.
KIOSK_STYLE = """
QMainWindow{
    background:#05080D;
}

QWidget{
    background:transparent;
    font-family:"Segoe UI";
    color:#C7DAE6;
}

QLabel{
    background:transparent;
    color:#C7DAE6;
}
"""
