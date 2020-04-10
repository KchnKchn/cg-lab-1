from PyQt5 import QtWidgets

from gui_impl.gui import GUI

def main():
    import sys
    app = QtWidgets.QApplication(sys.argv)
    gui = GUI()
    gui.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
