import sys

def split_words(string):
    return string.split(" ")

def split_sencences(string):
    return string.split(".")

def get_stats(filename):
    print("Analyzing {}".format(filename))
    with open(filename) as f:
        lines = f.readlines()
        print("\tlines:\t\t{}".format(len(lines)))
        wordcounts = dict()
        sentences_lengths = dict()
        wordcounter = 0
        sentencecounter = 0
        for line in lines:
            sentences = split_sencences(line)
            for sentence in sentences:
                sentencecounter += len(sentences)
                sentence_length = str(len(sentence))
                if sentence_length in sentences_lengths:
                    sentences_lengths[sentence_length] += 1
                else:
                    sentences_lengths[sentence_length] = 1
            words = split_words(line)
            wordcounter += len(words)
            for word in words:
                if word in wordcounts:
                    wordcounts[word] += 1
                else:
                    wordcounts[word] = 1

        print("\twords:\t\t{}".format(wordcounter))
        print("\tunique:\t\t{}, ({:%} off all words)".
              format(len(wordcounts), len(wordcounts) / wordcounter))
        print("\tsentences:\t{}".format(sentencecounter))



filename = sys.argv.pop()
get_stats(filename)
