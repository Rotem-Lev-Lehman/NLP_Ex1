import math
import re
from collections import Counter
from numpy.random import choice


class Ngram_Language_Model:
    """The class implements a Markov Language Model that learns a model from a given text.
        It supports language generation and the evaluation of a given string.
        The class can be applied on both word level and character level.
    """

    def __init__(self, n=3, chars=False):
        """Initializing a language model object.
        Args:
            n (int): the length of the markov unit (the n of the n-gram). Defaults to 3.
            chars (bool): True iff the model consists of ngrams of characters rather than word tokens.
                          Defaults to False
        """
        self.n = n
        self.chars = chars
        self.ngram_counter = None  # this counter will help us for the generation of the language model
        self.ngram_dictionary = None  # this dictionary will map from a context to the possible grams
        self.context_dictionary = None  # this dictionary will contain the context distribution (count)
        self.max_context_count = 0
        self.most_common_context = None
        self.ngram_most_common_dictionary = None  # this dictionary will return the most common gram for each context
        self.ngram_max_count_dictionary = None  # this dictionary will return the count of the most common gram for each context
        self.grams_set = None  # this set will hold all of the unique grams in the given text
        self.next_gram_choosing_dict = None  # this dictionary will hold for each context, both: (1) a list of possible next grams, and (2) a list of their corresponding probabilites

    def build_model(self, text):  # should be called build_model
        """populates a dictionary counting all ngrams in the specified text.

            Args:
                text (str): the text to construct the model from.
        """
        split_text = self.get_tokens(text)
        all_ngrams = self.get_all_ngrams(split_text)
        self.ngram_counter = Counter(all_ngrams)  # count each n-gram tuple and see how many times it occurred in the list.

        self.grams_set = set(split_text)

        self.ngram_dictionary = {}
        self.context_dictionary = {}
        self.max_context_count = 0
        self.ngram_most_common_dictionary = {}
        self.ngram_max_count_dictionary = {}
        self.next_gram_choosing_dict = {}
        for grams, num in self.ngram_counter.items():  # for each n-gram, map from the n-1 first grams to the last gram.
            grams_list = list(grams)
            grams_keys = tuple(grams_list[:-1])  # context of the n-gram
            gram_value = grams_list[-1]  # the n'th gram.

            # add to the ngram_dictionary:
            if grams_keys not in self.ngram_dictionary.keys():
                self.ngram_dictionary[grams_keys] = {}
            self.ngram_dictionary[grams_keys][gram_value] = num

            # check if need to add to ngram_most_common_dictionary:
            if grams_keys not in self.ngram_max_count_dictionary.keys() or self.ngram_max_count_dictionary[grams_keys] < num:
                self.ngram_max_count_dictionary[grams_keys] = num
                self.ngram_most_common_dictionary[grams_keys] = gram_value

            # add to the context_dictionary:
            if grams_keys not in self.context_dictionary.keys():
                self.context_dictionary[grams_keys] = 0
            self.context_dictionary[grams_keys] += num

            # check if we need to update the most common context:
            if self.context_dictionary[grams_keys] > self.max_context_count:
                self.max_context_count = self.context_dictionary[grams_keys]
                self.most_common_context = grams_keys

        for context, possible_grams in self.ngram_dictionary.items():
            total_possibilities = self.context_dictionary[context]
            only_grams = [x for x in possible_grams.keys()]
            prob_dist = [possible_grams[k] / total_possibilities for k in possible_grams.keys()]
            self.next_gram_choosing_dict[context] = (only_grams, prob_dist)

    def get_tokens(self, text):
        """Returns a list of tokens from the given text.
        Tokens are split either by space or to a list of characters, depending on the value of self.chars.

            Args:
                text (str): the text we wish to tokenize.

            Return:
                List. The list of tokens.

        """
        if self.chars:
            split_text = list(text)  # if we choose to use n-grams of characters, than we shall split each character to different list entry.
        else:
            split_text = text.split(' ')  # split the text by space, so each word will be in a different list entry.
        return split_text

    def get_all_ngrams(self, split_text):
        """Returns a list of n-grams from the given split text.

                    Args:
                        split_text (list): the list of all tokens.

                    Return:
                        List. The list of all n-grams.

        """
        return zip(*[split_text[i:] for i in range(self.n)])  # each n-gram will be in a different tuple in the new list.

    def get_model(self):
        """Returns the model as a dictionary of the form {ngram:count}
        """
        return dict(self.ngram_counter.items())

    def generate(self, context=None, n=20):
        """Returns a string of the specified length, generated by applying the language model
        to the specified seed context. If no context is specified the context should be sampled
        from the models' contexts distribution. Generation should stop before the n'th word if the
        contexts are exhausted.

            Args:
                context (str): a seed context to start the generated string from. Defaults to None
                n (int): the length of the string to be generated.

            Return:
                String. The generated text.

        """
        if context is not None:
            normalized_context = normalize_text(context)
            split_context = self.get_tokens(normalized_context)
            curr_context = tuple(split_context)
        else:  # context is None:
            curr_context = self.most_common_context

        curr_context_list = list(curr_context)
        sentence_grams = []
        for gram in curr_context_list:
            sentence_grams.append(gram)

        used_contexts = set()
        for i in range(n):
            # get next gram in the sentence:
            try:
                next_gram = self.get_next_gram(curr_context)
            except Exception:  # if we have never seen this context before, return the sentence as it is until now.
                break
            sentence_grams.append(next_gram)

            # mark the current context as "used" (that means that we have exhausted it):
            used_contexts.add(curr_context)

            # fix current context:
            del curr_context_list[0]
            curr_context_list.append(next_gram)
            curr_context = tuple(curr_context_list)

            # check if we have already exhausted the new context (and if so - stop building the sentence):
            #if curr_context in used_contexts:
            #    break

        return " ".join(sentence_grams)  # create a string from the list of the grams in the sentence

    def get_next_gram(self, context, sample_from_dist=True):
        """Returns the next gram using the given context.

                   Args:
                       context (tuple): the context that we want to find the next gram from.
                       sample_from_dist (bool): indicates if we want to use a distribution to select the next gram (True), or just take the most common completion (False). Defaults to True.

                   Returns:
                       String. The next gram we chose.
        """
        if sample_from_dist:
            return self.sample_next_gram(context)
        else:
            return self.get_most_common_gram(context)

    def sample_next_gram(self, context):
        """Returns the next gram using the given context. We choose the next gram using sampling from a distribution.

                           Args:
                               context (tuple): the context that we want to find the next gram from.

                           Returns:
                               String. The next gram we chose.
        """
        possible_ngrams, prob_dist = self.next_gram_choosing_dict[context]
        return choice(possible_ngrams, size=1, p=prob_dist)[0]

    def get_most_common_gram(self, context):
        """Returns the next gram using the given context. We choose the next gram to be the most common gram completion.

                                   Args:
                                       context (tuple): the context that we want to find the next gram from.

                                   Returns:
                                       String. The next gram we chose.
        """
        return self.ngram_most_common_dictionary[context]

    def evaluate(self, text):
        """Returns the log-likelihod of the specified text to be generated by the model.
           Laplace smoothing should be applied if necessary.

           Args:
               text (str): Text to ebaluate.

           Returns:
               Float. The float should reflect the (log) probability.
        """
        nt = normalize_text(text)
        split_text = self.get_tokens(nt)
        all_ngrams = self.get_all_ngrams(split_text)

        # go over all ngrams, and for each ngram, calculate the probability of it appearing.
        # multiply all of those probabilities. We can do that because of the Markov chain rule:
        probability = 0
        for ngram in all_ngrams:
            probability += math.log(self.smooth(" ".join(list(ngram))))  # we send it as a string because it is a requirement of the function "smooth"
        return probability  # we need to return the *log* likelihood...

    def smooth(self, ngram):
        """Returns the smoothed (Laplace) probability of the specified ngram.

            Args:
                ngram (str): the ngram to have it's probability smoothed

            Returns:
                float. The smoothed probability.
        """
        ngram_list = ngram.split(' ')
        context = tuple(ngram_list[:-1])
        last_gram = ngram_list[-1]
        V = len(self.grams_set)
        C_context = self.context_dictionary[context]
        C_ngram = self.ngram_dictionary[context][last_gram]
        P_laplace = (C_ngram + 1) / (C_context + V)  # the formula for the Laplace smoothing
        return P_laplace


def normalize_text(text, lower_text=True, remove_punctuations=True):
    """Returns a normalized string based on the specifiy string.
       You can add default parameters as you like (they should have default values!)
       You should explain your decitions in the header of the function.

       Args:
           text (str): the text to normalize
           lower_text (bool): specifies if we want to lower-case the given text. Defaults to True
           remove_punctuations (bool): specifies if we want to remove punctuations from the given text. Defaults to True

       Returns:
           string. the normalized text.
    """
    edited_text = text
    if lower_text:
        edited_text = text.lower()
    if remove_punctuations:
        edited_text = re.sub(r"""
                                [,.;@#?!&$]+  # Accept one or more copies of punctuation
                                \ *           # plus zero or more copies of a space,
                                """,
                             " ",          # and replace it with a single space
                             edited_text, flags=re.VERBOSE).rstrip()
    return edited_text


def who_am_i():
    """Returns a ductionary with your name, id number and email. keys=['name', 'id','email']
        Make sure you return your own info!
    """
    return {'name': 'Rotem Lev Lehman', 'id': '208965814', 'email': 'levlerot@post.bgu.ac.il'}
