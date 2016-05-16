#! /usr/bin/python

from Tkinter import *
import tkFont
from PIL import Image, ImageTk
import subprocess
import sys

text_box = None
entry = None

def generate():
    p = subprocess.check_output('./sample.py')
    global text_box
    text_box.delete("1.0", END) # Clear text box
    text_box.insert(END, p)

def generate_init():
    global entry
    init_seq = entry.get().lower()
    if init_seq == '':
        return

    try:
        p = subprocess.check_output(['./sample.py', '--init_seq', init_seq, '-n', '20',
            '--length', '20'])

        new_p = ''
        for row in p.split('\n'):
            if init_seq in row:
                new_p = new_p + row + '\n'
        p = new_p
    except:
        p = 'Word not found in dictionary.'

    global text_box
    text_box.delete("1.0", END) # Clear text box
    text_box.insert(END, p)

def main():
    root = Tk()
    root.attributes("-fullscreen", True)
    root.resizable(0,0)
    root.wm_title("Generating Headlines - Demo")

    # Font used in the GUI
    guiFont = tkFont.Font(family='Helvetica', size=24) #, weight='bold')

# ------ First frame ------
    fm = Frame(root)

    # Input box label
    init_label = Label(fm, text="Initial sequence", font=guiFont)
    init_label.pack(side=TOP, fill=X, padx=25, pady=(10,0))

    # Input box
    global entry
    entry = Entry(fm, font=guiFont)
    entry.pack(side=TOP, fill=X, padx=25)

    # Button for generation with initial sequence
    button_init = Button(fm, text='Generate', font="guiFont", height=2, command=generate_init)
    button_init.pack(side=TOP, fill=X, padx=25, pady=(0,0))

    # Button for random generation
    button_rand = Button(fm, text='Random', font="guiFont", height=2, command=generate)
    button_rand.pack(side=TOP, fill=X, padx=25, pady=(10,0))

    # Image
    image = Image.open("neural.png")
    image = image.resize((350, 350), Image.ANTIALIAS)
    photo = ImageTk.PhotoImage(image)

    label = Label(fm, image=photo)
    label.image = photo
    label.pack(side=BOTTOM, fill=X, padx=25)

    fm.pack(side=LEFT, fill=BOTH)

# ------ Second frame ------
    fm2 = Frame(root)

    # Text box for the generated results
    global text_box
    text_box = Text(fm2)
    text_box.config(font=guiFont)
    text_box.pack(fill=BOTH, expand=1)

    fm2.pack(side=LEFT, expand=1, fill=BOTH)

    root.mainloop()

if __name__ == "__main__":
     main()
