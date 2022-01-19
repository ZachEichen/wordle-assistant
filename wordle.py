import json 

class Wordle: 
    def __init__(self):
        json_obj = None
        with open('wordle_words.json','r') as file :
            json_obj = json.load(file)
            
        self.words = json_obj['words']
        self.sc_words = json_obj['sc_words']
        self.nltk_words = json_obj['nltk_words']