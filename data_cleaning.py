"""This script loops over the html files in a folder, 
extracts only the text data, and splits it into sentences.
The sentences are output in a tsv file for further annotation. """

import glob
import re

from bs4 import BeautifulSoup

HTML_DATA_FILEPATH = "data_html/*.html"
DELIMITERS = r"[。？！\n]"    # Delimiters for spliting the text data into sentences

with open("sentences_to_annotate.tsv", "w", encoding="utf-8") as output_file:
    print("text\tlabel", file=output_file)    # Print the output TSV file header
    # Loop over each html file
    for filepath in glob.glob(HTML_DATA_FILEPATH):
        with open(filepath, "r", encoding="utf-8") as source_html:
            # Extract the text data
            text_data = BeautifulSoup(source_html.read(), 'html.parser').get_text(separator="\n", strip=True)
            text_data = re.sub(r"[「」]", "", text_data)    # Removing the quotes
            # Split the text data into sentences
            lines = re.split(DELIMITERS, text_data)
            for line in lines[2:]:    # Skip the first two lines (the headline)
                if line:    # Skip the empty lines
                    print(f"{line}\t", file=output_file)