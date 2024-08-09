import json
import os
import nltk
from nltk.parse import stanford
import stanza 
# stanza.download('en',model_dir='stanza_resources')
import subprocess
stanza.install_corenlp()
# from nltk.stem import WordNetLemmatizer
# from nltk.tokenize import word_tokenize
# from nltk.tokenize import sent_tokenize
# from nltk.corpus import stopwords
import speech_recognition as sr
from nltk.tree import Tree, ParentedTree
from nltk.parse.corenlp import CoreNLPParser
from nltk.tree import *
from six.moves import urllib
import zipfile
import sys
import time
import ssl
from nltk.tokenize import word_tokenize
from nltk import pos_tag
import string

ssl._create_default_https_context = ssl._create_unverified_context
from flask import Flask,request,render_template,send_from_directory,jsonify

app =Flask(__name__,static_folder='static', static_url_path='')

import stanza
# from stanza.server import CoreNLPClient
import pprint 

# These few lines are important
BASE_DIR = os.path.dirname(os.path.realpath(__file__))


# Set the JAVAHOME environment variable
os.environ['JAVAHOME'] = r'C:\Program Files\Java\jdk-22'
# Ensure the JAVA path is included in the PATH environment variable
os.environ['PATH'] += os.pathsep + r'C:\Program Files\Java\jdk-22\bin'
os.environ['NLTK_DATA'] = '/usr/local/share/nltk_data/'
# Set the CORENLP_HOME environment variable
os.environ["CORENLP_HOME"] = r"C:\Users\madhe\stanza_corenlp"

def start_corenlp_server():
    corenlp_dir = os.environ["CORENLP_HOME"]
    
    # Start the CoreNLP server
    server = subprocess.Popen([
        'java', '-mx4g', '-cp', os.path.join(corenlp_dir, '*'),
        'edu.stanford.nlp.pipeline.StanfordCoreNLPServer',
        '-port', '9000', '-timeout', '60000'],
        cwd=corenlp_dir)
    
    # Give the server some time to start
    time.sleep(10)
    
    return server

# Start the server when the script is run
corenlp_server = start_corenlp_server()

   

def reporthook(count, block_size, total_size):
    global start_time
    if count == 0:
        start_time = time.perf_counter()
        return
    duration = time.perf_counter() - start_time
    progress_size = int(count * block_size)
    speed = int(progress_size / (1024 * duration))
    percent = min(int(count*block_size*100/total_size),100)
    sys.stdout.write("\r...%d%%, %d MB, %d KB/s, %d seconds passed" %
                    (percent, progress_size / (1024 * 1024), speed, duration))
    sys.stdout.flush()



# Pipeline for stanza (calls spacy for tokenizer)
en_nlp = stanza.Pipeline('en', processors={'tokenize': 'spacy', 'ner': 'spacy'})	
# print(stopwords.words('english'))

# stop words that are not to be included in ISL
stop_words = set(["am","are","is","was","were","be","being","been","have","has","had","the","to","some","by",
					"does","did","could","should","would","can","shall","will","may","might","must","let",'for','a',','])

# sentences array
sent_list = []
# sentences array with details provided by stanza
sent_list_detailed=[]

# word array
word_list=[]

# word array with details provided by stanza 
word_list_detailed=[]

# converts to detailed list of sentences ex. {"text":"word","lemma":""}
def convert_text_to_lists(text):
    # Convert text to a detailed list of sentences
    for sentence in text.sentences:
        if sentence.text.strip():  # Check if sentence has non-empty content
            sent_list.append(sentence.text)
            sent_list_detailed.append(sentence)

    # Convert sentences to a list of words
    for sentence in sent_list_detailed:
        temp_list = []
        temp_list_detailed = []
        for word in sentence.words:
            if word.text.strip():  # Check if word has non-empty content
                temp_list.append(word.text)
                temp_list_detailed.append(word)
        word_list.append(temp_list.copy())
        word_list_detailed.append(temp_list_detailed.copy())
        temp_list.clear()
        temp_list_detailed.clear()

# Filter stop words from the word list
def filter_words(word_list):
    temp_list = []
    final_words = []
    for words in word_list:
        temp_list.clear()
        for word in words:
            if word.lower() not in stop_words:
                temp_list.append(word)
        final_words.append(temp_list.copy())
    for words in word_list_detailed:
        for i, word in enumerate(words):
            if words[i].text in stop_words:
                del words[i]
                break
    return final_words

# # Remove punctuation from the word lis
def remove_punc(word_list):
    punctuations = set(string.punctuation)

    for words in word_list:
        # Collect indices and words to remove
        indices_to_remove = []
        for i, word in enumerate(words):
            if word in punctuations:
                indices_to_remove.append(i)

        # Remove items from words based on collected indices
        for index in sorted(indices_to_remove, reverse=True):
            del words[index]

def remove_punct(word_list, word_list_detailed):
    for i, (words, words_detailed) in enumerate(zip(word_list, word_list_detailed)):
        # Create a list of indices to remove
        indices_to_remove = []
        words_to_remove = []

        # Collect indices and words to remove
        for j, (word, word_detailed) in enumerate(zip(words, words_detailed)):
            if isinstance(word_detailed, dict) and word_detailed.get('upos') == 'PUNCT':
                indices_to_remove.append(j)
                words_to_remove.append(word_detailed['text'])

        # Remove items from words_detailed and words based on collected indices
        for index in sorted(indices_to_remove, reverse=True):
            del words_detailed[index]

        for word in words_to_remove:
            if word in words:
                words.remove(word)


# Lemmatize words
def lemmatize(final_word_list):
    for words, final in zip(word_list_detailed, final_word_list):
        for i, (word, fin) in enumerate(zip(words, final)):
            if fin in word.text:
                if len(fin) == 1:
                    final[i] = fin
                else:
                    final[i] = word.lemma
    for word in final_word_list:
        print("final_words", word)

# Label parse subtrees
def label_parse_subtrees(parent_tree):
    tree_traversal_flag = {}
    for sub_tree in parent_tree.subtrees():
        tree_traversal_flag[sub_tree.treeposition()] = 0
    return tree_traversal_flag

# Handle noun clauses in the tree
def handle_noun_clause(i, tree_traversal_flag, modified_parse_tree, sub_tree):
    if tree_traversal_flag[sub_tree.treeposition()] == 0 and tree_traversal_flag[sub_tree.parent().treeposition()] == 0:
        tree_traversal_flag[sub_tree.treeposition()] = 1
        modified_parse_tree.insert(i, sub_tree)
        i += 1
    return i, modified_parse_tree

# Handle verb/preposition clauses in the tree
def handle_verb_prop_clause(i, tree_traversal_flag, modified_parse_tree, sub_tree):
    for child_sub_tree in sub_tree.subtrees():
        if child_sub_tree.label() == "NP" or child_sub_tree.label() == 'PRP':
            if tree_traversal_flag[child_sub_tree.treeposition()] == 0 and tree_traversal_flag[child_sub_tree.parent().treeposition()] == 0:
                tree_traversal_flag[child_sub_tree.treeposition()] = 1
                modified_parse_tree.insert(i, child_sub_tree)
                i += 1
    return i, modified_parse_tree

# Modify the tree structure according to POS
def modify_tree_structure(parent_tree):
    tree_traversal_flag = label_parse_subtrees(parent_tree)
    modified_parse_tree = Tree('ROOT', [])
    i = 0
    for sub_tree in parent_tree.subtrees():
        if sub_tree.label() == "NP":
            i, modified_parse_tree = handle_noun_clause(i, tree_traversal_flag, modified_parse_tree, sub_tree)
        if sub_tree.label() == "VP" or sub_tree.label() == "PRP":
            i, modified_parse_tree = handle_verb_prop_clause(i, tree_traversal_flag, modified_parse_tree, sub_tree)
    for sub_tree in parent_tree.subtrees():
        for child_sub_tree in sub_tree.subtrees():
            if len(child_sub_tree.leaves()) == 1:
                if tree_traversal_flag[child_sub_tree.treeposition()] == 0 and tree_traversal_flag[child_sub_tree.parent().treeposition()] == 0:
                    tree_traversal_flag[child_sub_tree.treeposition()] = 1
                    modified_parse_tree.insert(i, child_sub_tree)
                    i += 1
    return modified_parse_tree

# converts the text in parse trees
def reorder_eng_to_isl(input_string):
	# count = 0
    # for word in input_string:
    #     if len(word) == 1:
    #         count += 1
    # if count == len(input_string):
    #     return input_string

    parser = CoreNLPParser(url='http://localhost:9000')
    possible_parse_tree_list = list(parser.parse(input_string))
    print("Possible parse trees:", possible_parse_tree_list)
    parse_tree = possible_parse_tree_list[0]
    parent_tree = ParentedTree.convert(parse_tree)
    print("xsflknfgkhbg")
    print(parent_tree)
    modified_parse_tree = modify_tree_structure(parent_tree)
    parsed_sent = modified_parse_tree.leaves()
    return parsed_sent



# final word list
final_words= []
# final word list that is detailed(dict)
final_words_detailed=[]

# def extract_word_lists(some_text):
#     # Ensure some_text is a string before tokenizing
#     if isinstance(some_text, stanza.Document):
#         some_text = some_text.text  # Extract text from the stanza Document object

#     words = word_tokenize(some_text)
#     word_detailed = pos_tag(words)
#     return words, word_detailed

# # pre processing text
def pre_process(text):
    # global word_list, word_list_detailed
    # word_list, word_list_detailed = extract_word_lists(text)
    # print(f"word_list: {word_list}")  # Debug print
    # print(f"word_list_detailed: {word_list_detailed}")  # Debug print
    remove_punct(word_list, word_list_detailed)
    final_words.extend(filter_words(word_list))
    lemmatize(final_words)


# checks if sigml file exists of the word if not use letters for the words
def final_output(input):
	final_string=""
	valid_words=open("words.txt",'r').read()
	valid_words=valid_words.split('\n')
	fin_words=[]
	for word in input:
		word=word.lower()
		if(word not in valid_words):
			for letter in word:
				# final_string+=" "+letter
				fin_words.append(letter)
		else:
			fin_words.append(word)

	return fin_words

final_output_in_sent=[]

# converts the final list of words in a final list with letters seperated if needed
def convert_to_final():
    for words in final_words:
        temp  = []
        for w in words:
            if w not in temp:
                temp.append(w)
        final_output_in_sent.append(final_output(temp))

    print("dsfjbgsdfhgsvrgtusivr tes")
    print(temp)
    print(final_output_in_sent)



# takes input from the user
def take_input(text):
    # Strip the input text and remove newlines and tabs
    test_input = text.strip().replace("\n", "").replace("\t", "")
    
    # Split the text into smaller chunks (sentences or segments)
    segments = [segment.strip() + '.' for segment in test_input.split('.') if segment.strip()]


    # Process each chunk with the en_nlp function and then convert function
    for segment in segments:
        processed_segment = en_nlp(segment)
        convert(processed_segment)



def convert(some_text):
    convert_text_to_lists(some_text)

	# reorders the words in input
    for i,words in enumerate(word_list):
        word_list[i]=reorder_eng_to_isl(words)

    print("ksjdbgdlbgkzsfbgbgvgb")
    print(some_text)
    print(word_list)

	# removes punctuation and lemmatizes words
    pre_process(some_text)
    convert_to_final()
    remove_punc(final_output_in_sent)
    print_lists()
	

def print_lists():
	print("--------------------Word List------------------------")
	pprint.pprint(word_list)
	print("--------------------Final Words------------------------")
	pprint.pprint(final_words)
	print("---------------Final sentence with letters--------------")
	pprint.pprint(final_output_in_sent)

# clears all the list after completing the work
def clear_all():
	sent_list.clear()
	sent_list_detailed.clear()
	word_list.clear()
	word_list_detailed.clear()
	final_words.clear()
	final_words_detailed.clear()
	final_output_in_sent.clear()
	


# dict for sending data to front end in json



@app.route('/',methods=['GET'])
def index():
	clear_all()
	return render_template('index.html')



@app.route('/',methods=['GET','POST'])
def flask_test():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'input_file' in request.files:
            
            print('**********************************************************')
            file = request.files.get('input_file')
            if file and file.filename != '':
                try:
                    text = file.read().decode('utf-8')
                    print("Text is:", text)
                except Exception as e:
                    print("Error reading file:", e)
                    return jsonify({"error": "Failed to read the file"}), 400
            else:
                text = request.form.get('text', '')
                print("Tex:", text)
        else:
            text = request.form.get('text', '')
            print("Text is:", text)
        

        text_list = text.split(".")
        text_list = [sentence.strip() for sentence in text_list if sentence.strip()]
        print("text_list")
        print(text_list)

        final_final_words_dict = {}
        no_of_words = 0

        for t in text_list:
            
            take_input(t)

            # fills the json
            final_words_dict = {}
            for words in final_output_in_sent:
                for i,word in enumerate(words,start=1):
                    if i:
                        final_words_dict[i]=word
                    else:   
                        clear_all()

            print("---------------Final words dict--------------")

            for key in final_words_dict.keys():
                if len(final_words_dict[key])==1:
                    final_words_dict[key]=final_words_dict[key].upper()
            print(final_words_dict)
            for k in final_words_dict:
                no_of_words += 1
                final_final_words_dict[no_of_words] = final_words_dict[k]

            clear_all()

    return jsonify(final_final_words_dict)

#############################################################################

################################################################################

# serve sigml files for animation
@app.route('/static/<path:path>')
def serve_signfiles(path):
    print(f"============================{path}==============================")
    print("here")
    return send_from_directory('static',path)


if __name__=="__main__":
    app.run(host='0.0.0.0')
