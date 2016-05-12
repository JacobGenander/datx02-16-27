#! /usr/bin/python

from Tkinter import *
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
    root.resizable(0,0)
    root.wm_title("Generating Headlines - Demo")

# ------ First frame ------
    fm = Frame(root)

    # Input box label
    init_label = Label(fm, text="Initial sequence")
    init_label.pack(side=TOP, fill=X, padx=25, pady=(10,0))

    # Input box
    global entry
    entry = Entry(fm)
    entry.pack(side=TOP, fill=X, padx=25)

    # Button for generation with initial sequence
    button_init = Button(fm, text='Generate', command=generate_init)
    button_init.pack(side=TOP, fill=X, padx=25, pady=(0,0))

    # Button for random generation
    button_rand = Button(fm, text='Random', command=generate)
    button_rand.pack(side=TOP, fill=X, padx=25, pady=(10,0))

    # Image
    image = Image.open("neural.png")
    image = image.resize((250, 250), Image.ANTIALIAS)
    photo = ImageTk.PhotoImage(image)

    label = Label(fm, image=photo)
    label.image = photo
    label.pack(side=TOP, fill=X, padx=25)

    fm.pack(side=LEFT, fill=BOTH)

# ------ Second frame ------
    fm2 = Frame(root)

    # Text box for the generated results
    global text_box
    text_box = Text(fm2, height=25, width=100)
    text_box.pack(fill=X)

    fm2.pack(side=LEFT, padx=10)

    root.mainloop()

if __name__ == "__main__":
     main()
