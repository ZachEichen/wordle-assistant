import json
from collections import Counter
import math
from typing import * 

# safe tqdm import
try:
    from tqdm import tqdm
except ImportError as e:
    tqdm = lambda x: x


class Wordle:
    def __init__(self):
        json_obj = None
        with open("wordle_words.json", "r") as file:
            json_obj = json.load(file)

        self.words = json_obj["words"]
        self.sc_words = json_obj["sc_words"]
        self.nltk_words = json_obj["nltk_words"]
        self.reset_guess_info()

    def reset_guess_info(self):
        self.known_letters = {}
        self.yellow_letters = []
        self.wrong_letters = ""
        self.possible_words = None

    def add_guess(self, guess: str, response: str):
        guess_known = {}
        guess_yellow = []
        for i in range(5):
            if response[i] == "g":
                guess_known[i] = guess[i]
            elif response[i] == "y":
                guess_yellow.append((i, guess[i]))
            else:
                if guess[i] not in guess_known.values() and guess[i] not in [
                    letter for _, letter in guess_yellow
                ]:
                    self.wrong_letters += guess[i]
        self.known_letters.update(guess_known)
        self.yellow_letters += guess_yellow

    def add_guesses(self, guesses_list, reset_info=False):
        if reset_info:
            self.reset_guess_info()
        for guess, resp in guesses_list:
            self.add_guess(guess, resp)

    def get_possible_words(self, word_pool="sourcecode words", verbose=False):
        if word_pool == "valid_words":
            self.possible_words = self.words
        elif word_pool == "nltk_words":
            self.possible_words = self.nltk_words
        else:  # default case: use sc words
            self.possible_words = self.sc_words

        # narrow all words by known letters first
        for ind, val in self.known_letters.items():
            if val == "":
                continue
            if verbose:
                print(f"narrowing based off of '{val}' in spot {ind}")
            self.possible_words = [
                word for word in self.possible_words if word[ind] == val
            ]
            if verbose:
                print(f"\t list size is {len(list(self.possible_words))}")
        # then narrow by yellow letters
        for ind, val in self.yellow_letters:
            if val == "":
                continue
            if verbose:
                print(f"narrowing based off of '{val}' in spot {ind}")
            self.possible_words = [
                word for word in self.possible_words if val in word and word[ind] != val
            ]
            if verbose:
                print(f"\t list size is {len(list(self.possible_words))}")
        for let in self.wrong_letters:
            if verbose:
                print(f"narrowing based off of incorrect letter '{let}'")
            self.possible_words = [
                word
                for word in self.possible_words
                if let not in word or let in self.known_letters.values()
            ]
            if verbose: print(f"list size is {len(self.possible_words)}")

        return self.possible_words

    def get_best_guesses(
        self,
        return_dict=True,
        return_arr = True, 
        verbose=False,
        hard_mode: Union[bool, str]="Default",
        use_entropy:bool=True,
        word_pool="valid_words",
        print_best_entropy:bool = True,
        use_tqdm:bool = True, 
        give_distribution = False,
        hard_threshold = 8 
        ):
        
        if hard_mode == 'Default': 
            hard_mode  = len(self.possible_words) <= hard_threshold
            print(f"using hard mode: {hard_mode}")
        elif type(hard_mode) == str: 
            raise  ValueError('invalid arg for hard_mode')
            
        possible_guesses = None
        if hard_mode: 
            possible_guesses = self.possible_words
        elif word_pool.lower() =="valid_words": 
            possible_guesses = self.words
        elif word_pool.lower() == 'source_words': 
            possible_guesses = self.sc_words()
        elif word_pool.lower() == "nltk_words": 
            possible_guesses = self.nltk_words()
        else: 
            raise  ValueError('invalid arg for word_pool')
        
        best_metric = 0 
        best_guesses = []
        maybe_tqdm = tqdm  if use_tqdm else lambda x: x 
        for guess in maybe_tqdm(possible_guesses):
            resps = []
            for ans in self.possible_words: 
                resp = ['g' if ans[i] == c else 'y' if c in ans else 'x' for i,c in enumerate(guess)]
                resps.append(''.join(resp))

            resp_counts = Counter(resps)
            num_uniques = len(resp_counts)
            entropy = _calc_shannon_entropy(resp_counts)
            metric = entropy if use_entropy else num_uniques

            if metric > best_metric: 
                best_guesses = []
                best_metric = metric

            if metric >= best_metric: 
                entr = {
                    "guess":        guess,
                    "num_classes":  num_uniques, 
                    "entropy bits": entropy
                }
                if give_distribution: 
                    entr["resp_distro"] =  resp_counts

                best_guesses.append(entr)
        if print_best_entropy and use_entropy: 
            print(f'best guess found gave {best_metric} bits of information')
            
        best_guesses.sort( key= lambda x: x['entropy bits'], reverse=True)
        if return_arr: 
            if return_dict:
                return best_guesses
            else: 
                return [ guess['guess'] for guess in best_guesses]
        else: # need to choose 'best' word to return 
            ret_guess = best_guesses[0]
            for guess in best_guesses: 
                if guess in self.possible_words: 
                    ret_guess = guess
                    break
            return ret_guess if return_dict else ret_guess['guess']
            
def _calc_shannon_entropy(counts: Counter):
    all_counts = [count for _, count in counts.items()]
    all_words = sum(all_counts)
    probs = (count / all_words for count in all_counts)
    return sum((-1 * prob * math.log(prob) for prob in probs))
