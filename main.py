import logging
import os

from PIL import ImageTk, Image, ImageDraw
import PIL
from tkinter import *
import tkinter as tk
from tkinter import ttk

from AnalysPhoto import analysPhoto, updateModel, answer
from createModel import createModel

width = 500
height = 200
center = height // 2
white = (255, 255, 255)
green = (0, 128, 0)
filename = "image.png"

class tkinterApp(tk.Tk):  # створення класу, який відповідає за усі фрейми
    container = None
    def __init__(self, *args, **kwargs):
        # __init__ function for class Tk
        tk.Tk.__init__(self, *args, **kwargs)

        self.container = tk.Frame(self)  # Створення фрейму
        self.container.pack(side="top", fill="both", expand=True)  # додаємо на екран

        self.frames = {}

        for F in (StartPage, Page1, Page2, Page3):  # Створення 5 фреймів та додання в змінну
            frame = F(self.container, self)
            self.frames[F] = frame
            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame(StartPage)  # Початкова сторінка

    def show_frame(self, cont):
        if self.frames[cont] == self.frames[Page3]:
            self.frames[cont] = Page3(self.container, self).grid(row=0, column=0, sticky="nsew")
        frame = self.frames[cont]
        frame.tkraise()



class StartPage(tk.Frame):  # Початкова сторінка
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = ttk.Label(self, text="Main menu", font=("Verdana", 35))
        label.place(relx=0.5, rely=0.2, anchor=CENTER)
        button1 = ttk.Button(self, text="Canvas",
                             command=lambda: controller.show_frame(Page1))  # Кнопка переходу на іншу сторінку

        button1.place(relx=0.5, rely=0.45, anchor=CENTER)

        button2 = ttk.Button(self, text="Load madel",
                             command=lambda: controller.show_frame(Page2))  # Кнопка переходу на іншу сторінку

        button2.place(relx=0.5, rely=0.65, anchor=CENTER)


class Page1(tk.Frame):  # Створення наступного фрейму
    cv = None
    image1 = None
    draw = None

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        # Tkinter create a canvas to draw on
        self.cv = Canvas(self, width=width, height=height, bg='white')
        self.cv.pack()
        # PIL create an empty image and draw object to draw on
        # memory only, not visible
        self.image1 = PIL.Image.new("RGB", (width, height), white)
        self.draw = ImageDraw.Draw(self.image1)
        # do the Tkinter canvas drawings (visible)
        # cv.create_line([0, center, width, center], fill='green')

        self.cv.pack(expand=YES, fill=BOTH)
        self.cv.bind("<B1-Motion>", self.paint)

        # do the PIL image/draw (in memory) drawings
        # draw.line([0, center, width, center], green)

        # PIL image can be saved as .png .jpg .gif or .bmp file (among others)
        # filename = "my_drawing.png"
        # image1.save(filename)
        button4 = ttk.Button(self,text="save", command=self.save)

        button3 = ttk.Button(self,text="clear canvas", command=self.clearCanvas)
        button4.pack()

        button3.pack()

        button11 = ttk.Button(self, text="Main menu",
                              command=lambda: controller.show_frame(StartPage))  # Кнопка переходу на іншу сторінку
        button11.place(relx=0.1, rely=0.85, anchor=CENTER)

        button21 = ttk.Button(self, text="Load model",
                              command=lambda: controller.show_frame(Page2))  # Кнопка переходу на іншу сторінку
        button21.place(relx=0.1, rely=0.95, anchor=CENTER)

    def clearCanvas(self):
        self.cv.delete('all')
        self.image1 = PIL.Image.new("RGB", (width, height), white)
        self.draw = ImageDraw.Draw(self.image1)

    def save(self):
        self.image1.save(filename)
        self.controller.show_frame(Page3)



    def paint(self, event):
        # python_green = "#476042"
        x1, y1 = (event.x - 1), (event.y - 1)
        x2, y2 = (event.x + 1), (event.y + 1)
        self.cv.create_oval(x1, y1, x2, y2, fill="black", width=5)
        self.draw.line([x1, y1, x2, y2], fill="black", width=5)


class Page2(tk.Frame):  # Сторінка з читанням бд
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = ttk.Label(self, text="Read", font=("Verdana", 35))
        label.pack()

        button2 = ttk.Button(self, text="create model", command=createModel)
        button2.place(relx=0.5, rely=0.6, anchor=CENTER)

        button5 = ttk.Button(self, text="Canvas",
                             command=lambda: controller.show_frame(Page1))  # Кнопка переходу на іншу сторінку

        button5.place(relx=0.1, rely=0.85, anchor=CENTER)

        button6 = ttk.Button(self, text="Main menu",
                             command=lambda: controller.show_frame(StartPage))  # Кнопка переходу на іншу сторінку

        button6.place(relx=0.1, rely=0.95, anchor=CENTER)


class Page3(tk.Frame):  # Початкова сторінка
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = ttk.Label(self, text="Result", font=("Verdana", 35))
        label.pack()
        Answer = ttk.Label(self, text= analysPhoto(filename), font=("Verdana", 12))
        Answer.pack()
        Answer = ttk.Label(self, text='If answer uncorrect write line and submit', font=("Verdana", 12))
        Answer.pack()
        self.inputField = tk.Entry(self)
        self.inputField.pack()

        button7 = ttk.Button(self, text="Submit",
                             command=self.updateModel)  # Кнопка переходу на іншу сторінку

        button7.place(relx=0.5, rely=0.65, anchor=CENTER)
        button6 = ttk.Button(self, text="Main menu",
                             command=lambda: controller.show_frame(StartPage))  # Кнопка переходу на іншу сторінку

        button6.place(relx=0.5, rely=0.85, anchor=CENTER)

    def updateModel(self):
        print(self.inputField.get())
        updateModel(self.inputField.get())

    def refresh(self):
        self.destroy()
        self.__init__()

app = tkinterApp()
app.mainloop()
