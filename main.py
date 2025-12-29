# main.py
import sys
from PyQt5.QtWidgets import QApplication
from coach_window import VirtualCoachWindow

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = VirtualCoachWindow()
    win.show()
    sys.exit(app.exec_())