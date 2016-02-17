import numpy as np


# Had to supply a converter to the `genfromtext`-function to get unicode
def string_converter(s):
    return s.decode("utf-8")


# Reads a glove file and returns an numpy array where the first column
# contains the words and columns > 0 contains the values in each dimension
def read_glove(input_file, dimensions):

    # The format for the words: Unicode, up to 128 characters
    columns_datatype = [("word", np.dtype("U128"))]

    # Create the datatypes for the dimensional data: 8 byte floats
    for i in range(dimensions):
        columns_datatype.append(("dim{}".format(i), np.dtype("f8")))

    # print("Using the following datatypes for the columns\n")
    # print(columns_datatype)

    glove_data = np.genfromtxt(
        fname=input_file,
        dtype=columns_datatype,
        converters={0: string_converter},
        comments=None,
        delimiter=' ',
        loose=False
    )
    return glove_data


# Reads a file of newline-separated strings
def read_lines(input_file):
    headlines_lines = np.genfromtxt(
        fname=file_text,
        dtype=np.dtype("str"),
        converters={0: string_converter},
        delimiter="\n"
    )
    return headlines_lines


# TODO: Implement this using numpy (This function serves as an example only, it
# is slow as ****)
# I think that the best approach is to use a dict, as it uses a hash table
# internally, which should speed things up a bit.
def glovify_text(glove_data, text):
    print("Translating {} to glove vectors".format(text))
    for word in text:
        for key in glove_data:
            if np.core.defchararray.equal(key[0], word):
                print(key)


file_glove = "/mnt/ml_data/glove.6B.50d.txt"
file_text = "/mnt/ml_data/headlines.test.txt"

glove_dimensions = 50
glove_data = read_glove(file_glove, glove_dimensions)

print("Read {} vectors".format(glove_data.size))

headlines_lines = read_lines(file_text)
print("Read {} headlines".format(headlines_lines.size))


all_words = []
for line in headlines_lines:
    line_words = line.split(" ")
    all_words.extend(line_words)

print("Got {} words from headlines".format(len(all_words)))

words, counts = np.unique(all_words, return_counts=True)
counts_words = np.rec.fromarrays((counts, words), names=('counts', 'words'))

print("Unique words: {} ({:%} of all words)".format(
    len(counts_words),
    len(counts_words)/len(all_words))
)

counts_words.sort(axis=0)
print(counts_words)

test = headlines_lines[321].split(" ")
glovify_text(glove_data, test)
