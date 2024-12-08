from PySide6.QtWidgets import (
    QMainWindow,
    QApplication,
    QStatusBar,
    QBoxLayout,
    QPushButton,
    QLabel,
    QToolBar,

)


class ui(QMainWindow):
    def __init__(self):
        super.__init__(self)
        self.layout(QBoxLayout())
        self.setWindowTitle("ImageClassifer")





if __name__ == "__main__":

    app = QApplication()
    ui().show()
    app.exec()