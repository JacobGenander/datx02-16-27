% Working with LyX, git and other tools
% Rickard Lantz
% 2016-02-01


This document gives a brief introduction to LyX, and describes some good practices when using LyX together with git.

I also recommend some tools that I use when writing documents


## What is LyX

LyX is a document editor which follows the "LaTeX-way" of writing. In a nutshell, this means that LaTeX decides how the content should be formatted, what layout to use etc.
Instead the writer will focus on what the text *is*. For example, if you want to put emphasis on text in Microsoft Word, you would probably alter the text style to be **bold** or *italics*. In latex you would use the command `\emph{my text}`. How the text will look in the produced document is then decided by the document class.
This way, the documents get a very consistent and uniform appearance.

LyX is a tool which allows you to write documents according to the "What You See Is What You Mean" philosophy. And it also provides tools to give a visual representation of for example maths and figures, and provides spell-checking.
Note however, that these representations in LyX are not the same ones that will be seen in the compiled document, the viewport in LyX is a totally separate entity from the LaTeX-compiler, it merely understands and uses the same language.

LyX also provides GUI-tools through which many document-specific-preferences, such as document classes, extra LaTeX-packages. margin-settings, word-, line- and paragraph spacings etc. can be configured. There is also a way of specifying a traditional LaTeX preamble, but this should only be used for features which LyX does not support natively.

### The LyX format

Although LyX was intended to be a tool to write LaTeX documents, it is not a *LaTeX-editor*. This means that it does not read `.tex`-files.
Instead it uses it's own format, saved in `.lyx`-files^[This format is not compatible (interchangeable) with LaTeX, although LaTeX-files can be converted to LyX-files]

The format is text-based, but editing it by outside of LyX is strongly discouraged, and not a pleasant experience overall.


## Using LyX

The first thing to do is to read the excellent included help.

I would recommend at least reading the introduction that can be found from inside LyX (`Help -> Introduction`) (This should appear when you first start LyX).

In the same menu you can find the more detailed "User's Guide".


## LyX and Git

As described above, the LyX-format is text based, and it also uses a lot of newlines. This makes it quite a good contender for use with git.

Although a relatively painless process overall, LyX-files in a git-repository puts somewhat more emphasis on a good git workflow --- one do not want to create merge conflicts, as this would require manual handling of the LyX-files.

These are some pointers:

Pull often
:	Before you make any changes, make sure you are workning with the latest version of the document.

Commit often
:	When editing things, keep different types of edit in different commits. For example, if you do three thigs: fix spelling mistakes; change the position of an image; and add tom text to a section, these changes should each be in thier own separate commit.

Push often
:	When you have changes something, immeadiately pull down the latest version from the server to merge in you changes, and then push them up again.

Use descriptive commit messages
:	One does not to write an essay everytime you commit, but "add text" is for example somewhat to abigious. "add text to method" however, is much better.


# Other tools

LyX (or LaTeX for that matter) is not a complete tool-set for producing documents, and supplementary tools must be used for certain tasks.

## References and citations (BibTex)

LyX has support for inserting and previewing citations in documents using BibTeX-databses. However, LyX supplies no tools to create these databases.

I tend to either write them by hand, or by using the tool JabRef.

[JabRef](http://www.jabref.org/) is a Java-program which allows you to creates and organizes references. It also supplies tools to easily convert non-ASCII-characters to LaTeX-commands^[BibTeX is quite ancient, and has terrible support for special characters such as "å, ä, ö" etc. These characters must be converted to special LaTeX sequences before they can be included in the bibliography].

## Illustrations

There are many programs to create images and illustrations, each with their own pros and cons.

My general approach is to use vector graphics as much as possible. I also prefer text based formats, as they can often be edited and version controlled easily.

Incscape
:	Excellent program for creating svg-images.

Ipe
:	Program which exceels at mathematical- and physical illustrations. Has a very powerful snapping system, but is quite difficult to use. Have some wierd quirks when it comes to the image format.

TikZ
:	LaTeX-based language where you "code" the illustrations. Steep learning curve and therefore a slow workflow (at least for me), but extremely powerfull.

