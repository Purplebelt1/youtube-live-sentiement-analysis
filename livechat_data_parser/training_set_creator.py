import pandas as pd
import sys
import os
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QWidget, QHBoxLayout
from PyQt5.QtGui import QFont



from reader import load_dataframe

from IPython.display import display, HTML
import tkinter as tk


class App(QMainWindow):
    def __init__(self, df, output, index = 1):
        super().__init__()

        self.setWindowTitle("Model Training Dataframe Creator")
        self.setGeometry(100, 100, 600, 300)

        self.setStyleSheet("background-color: #F0F0F0;")  # Set a background color for the main window

        self.df = df

        self.output = output

        self.index = index - 1
        self.update_index()

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.central_widget.setStyleSheet("background-color: #FFFFFF; border: 1px solid #CCCCCC;")

        self.layout = QVBoxLayout()

        self.label = QLabel(self.df["content.message"][self.index], self)
        self.label.setWordWrap(True)
        self.label.setFont(QFont("Helvetica", 14))
        self.label.setMargin(10)
        self.layout.addWidget(self.label)

        self.button_layout = QHBoxLayout()
        button_css = "color: white; " +\
            "margin: 1\% 1\% 1\% 1\%; " +\
            "padding: 5px 10px 5px 10px" 

        self.very_pos_button = QPushButton("V. Pos", self)
        self.very_pos_button.clicked.connect(lambda: self.set_sentiment(1))
        self.very_pos_button.setStyleSheet("background-color: #00FF00; " + button_css)
        self.button_layout.addWidget(self.very_pos_button)

        self.pos_button = QPushButton("Pos", self)
        self.pos_button.clicked.connect(lambda: self.set_sentiment(0.5))
        self.pos_button.setStyleSheet("background-color: #90EE90;" + button_css)
        self.button_layout.addWidget(self.pos_button)

        self.neu_button = QPushButton("Neu", self)
        self.neu_button.clicked.connect(lambda: self.set_sentiment(0))
        self.neu_button.setStyleSheet("background-color: #D3D3D3; " + button_css)
        self.button_layout.addWidget(self.neu_button)

        self.neg_button = QPushButton("Neg", self)
        self.neg_button.clicked.connect(lambda: self.set_sentiment(-0.5))
        self.neg_button.setStyleSheet("background-color: #FF6347; " + button_css)
        self.button_layout.addWidget(self.neg_button)

        self.very_neg_button = QPushButton("V. Neg", self)
        self.very_neg_button.clicked.connect(lambda: self.set_sentiment(-1))
        self.very_neg_button.setStyleSheet("background-color: #8B0000; " + button_css)
        self.button_layout.addWidget(self.very_neg_button)

        self.layout.addLayout(self.button_layout)
        self.central_widget.setLayout(self.layout)

    def set_sentiment(self, value):
        self.df.loc[self.index, "sentiment"] = value
        self.update_index()
        self.update_label()

    def update_label(self):
        if self.index < len(self.df):
            self.label.setText(self.df["content.message"][self.index])
        else:
            self.label.setText("No more rows")

    def update_index(self):
        try:
            self.index = self.update_index_aux()
        except IndexError:
            print("\nReached end of dataset.\n")

    def update_index_aux(self, check_index=None):
        if check_index is None:
            check_index = self.index

        if check_index >= len(self.df):
            raise IndexError("Reached End of Data")
        

        try:
            if pd.isna(self.df["sentiment"][check_index]):
                return check_index
        except KeyError:
            return self.update_index_aux(check_index + 1)


        return self.update_index_aux(check_index + 1)

    def get_dataframe(self):
        return self.df
    
    def closeEvent(self, event):
        self.df.to_csv(self.output)
        print("Window Closed")


def create_set_sentiment_gui(input_file, output_file_name):
    if input_file[-4:] == "json":

        df = load_dataframe(input_file)

        sentiment = [None for i in range(len(df))]

        df["sentiment"] = sentiment

    elif input_file[-3:] == "csv":
        df = pd.read_csv(input_file)

    app = QApplication(sys.argv)
    window = App(df, output_file_name, 2500)
    window.show()


    sys.exit(app.exec_())


def main():
    #input = "C:/Users/William/youtube-livechat-scraper-main/TestJson_My super bowl prediction was right_1707846808.5228631.json"
    #input = "C:/Users/William/youtube-livechat-scraper-main/STORYTIME THEN BALDURS GATE_1707921673.599726.json"
    input = "yt_live_comments/second_training.csv"
    output = "yt_live_comments/second_training.csv"

    create_set_sentiment_gui(input, output)
    


if __name__ == "__main__":
    main()