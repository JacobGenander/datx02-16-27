import numpy as np
import time


# TODO: Determine whether the following string manipulations would improve the
# translations.
# * Use functions that strips sentences of special characters like ",.' etc.
# * Convert all sentences to lowercase


glove_dtype = np.dtype("f8")


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


def glove_data_to_dict(glove_data, dimensions):
    index_words = "word"
    index_vectors = []
    for dim in range(dimensions):
        index_vectors.append("dim{}".format(dim))

    # print(index_words)
    # print(index_vectors)

    words = glove_data[index_words]
    vectors = glove_data[index_vectors]

    glove_dict = dict()

    for i in range(glove_data.size):
        glove_dict[words[i]] = vectors[i]

    # print(words)
    # print(vectors)

    return glove_dict


def glovify_words(glove_dict, words, dimensions):
    # Get the array type from
    vec_repr = np.empty([len(words), dimensions], dtype=glove_dtype)
    for i, word in enumerate(words):
        vector = glove_dict.get(word, None)
        if vector is not None:
            print("{}:\t{}".format(word, vector))
            # TODO: Figure out how to replace row in array. Combining normal
            # numpy arrays and record arrays seems to create some problems as
            # one type is indexed with strings (keys) and the other one with
            # numerical indexes
            vec_repr[1,:] = vector
        else:
            print("Word {} not found".format(word))
            vec_repr[i] = None
    return vec_repr


# This function is implemented above using dicts
# (This function can still serve as a validation, it is however slow as ****)
def glovify_words_example(glove_data, words):
    print("Translating {} to glove vectors".format(words))
    for word in words:
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

print("Translating \"{}\" using slow as **** example version".format(test))
timer = time.time()
glovify_words_example(glove_data, test)
print("Sentence looked up in {} seconds".format(time.time() - timer))

print("Building dictionary...")
timer = time.time()
glove_dict = glove_data_to_dict(glove_data, glove_dimensions)
print("Dictionary built in {} seconds".format(time.time() - timer))

print("Translating...")
timer = time.time()
glovify_words(glove_dict, test, glove_dimensions)
print("Sentence looked up in {} seconds".format(time.time() - timer))
