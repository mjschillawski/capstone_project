# What's For Dinner?
# Authors: Michael Schillawski
# Data Science Immersive, General Assembly
# April 10, 2018
# Contact: mjschillawski@gmail.com

# Imports

import os
import json
import re
import string
import multiprocessing
import pickle

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pandas.io.json import json_normalize
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from joblib import Parallel, delayed

print(os.getcwd())

# Load Data

path = '/Users/mjschillawski/Desktop/Miscellaneous Data/Yummly28K/'
file = 'data_records_27638.txt'

data = pd.read_table(path+file,header=None,names=['recipe'],index_col=1)

path = '/Users/mjschillawski/Desktop/Miscellaneous Data/Yummly28K/metadata27638/'

recipes = []

for i in data.index:
    num = str(i)
    while len(num) < 5:
        num = '0' + num
        
    # https://stackoverflow.com/questions/28373282/how-to-read-a-json-dictionary-type-file-with-pandas
    with open(path+'meta'+num+'.json') as json_data:
        recipe = json.load(json_data)
        recipes.append(recipe)

recipes = json_normalize(recipes)   
recipes.to_csv('assets/recipes_dataset.csv')

recipes = pd.read_csv('assets/recipes_dataset.csv',index_col=0)

# transform ingredient field back into list when importing from CSV
recipes['ingredientLines'] = recipes['ingredientLines'].apply(
    lambda x: [item for item in x.split('\'') if item not in ('\,','[',']',', ')])

short_recipes = recipes[['attributes.course','attributes.cuisine','name','ingredientLines']]
short_recipes.to_csv('assets/short_recipes.csv')

# Natural Language Processing of Ingredients

# multi-threaded
def multi_process_ingredients(recipes,join=1,nondescript=0,drop_words=None,pantry=0,pantry_items=None):
    # create the patternsu

    # import punctuation characters
    # to remove all punctuation
    punct = string.punctuation
    punct_pattern = r"[{}]".format(punct)

    # to remove all numbers
    number_pattern = r"\d+\s"

    # embedded numbers
    embed_num_pattern = r".\d+."
    
    # removed prep methods
    prep_pattern = r"[a-z]+ed"
    
    # strip pluralization
    plural_pattern = r"s\s"
    
    # strip -ly
    ly_pattern = r"[a-z]+ly"
    
    # strip lead number
    lead_pattern = r"\d+[a-z]+"
    lead_repl = r"[a-z]+"
    
    # trail number
    trail_pattern = r"[a-z]+\d+"
    trail_repl = r"[a-z]+"
    
    recipes_ingredients = []
    ingredients = []

    for item in recipes:

        # strip punctuation
        text = re.sub(punct_pattern," ",item)
        # strip standalone numbers
        text = re.sub(number_pattern,"",text)
        # strip embedded numbers
        text = re.sub(embed_num_pattern,"",text)
        # strip preparation methods
        text = re.sub(prep_pattern,"",text)
        # strip pluralization
        text = re.sub(plural_pattern," ",text)
        # strip ly
        text = re.sub(ly_pattern,"",text)
        # lead
        text = re.sub(lead_pattern,lead_repl,text)
        # trail
        text = re.sub(trail_pattern,trail_repl,text)

        # tokenize
        tokenizer = RegexpTokenizer(r'\w+')
        processed_text = tokenizer.tokenize(text)

        # remove stop words
        processed_text = [text.lower() for text in processed_text if text.lower() 
                          not in stopwords.words('english')]
        
        # minimum word length
        processed_text = [text for text in processed_text if len(text) > 2]

        # remove non-descript recipe words
        if nondescript == 1 and drop_words != None:
            processed_text = [text.lower() for text in processed_text if text.lower()
                             not in drop_words]
            
        # remove pantry items
        if pantry == 1 and pantry_items != None:
            processed_text = [text.lower() for text in processed_text if text.lower()
                             not in pantry_items]

        # append all each list that to describe an ingredient of the recipe
        ingredients.append(processed_text)

    # joined space-separated strings
    # attach all modifiers that describe each ingredient (non-separated)
    clean_ingredients = [" ".join(word) for word in ingredients]

    # append all ingredients for each recipe
    recipes_ingredients.append(clean_ingredients)    
    
    if join == 0:
        pass
    else:
        recipes_ingredients = [" ".join(ingredient) for ingredient in recipes_ingredients]
    
    return recipes_ingredients


# initial process ingredients
num_cores = multiprocessing.cpu_count()
inputs = short_recipes['ingredientLines']

if __name__ == "__main__":
    recipes = Parallel(n_jobs=num_cores)(delayed(multi_process_ingredients)(i) for i in inputs)

# 1 list of ingredients for each recipe
recipes = [" ".join(recipe) for recipe in recipes]
recipes = pd.DataFrame(recipes)

# Word Counts for Stop Word Dictionary
# word counts
# get the words that occur most often in recipes
# these are candidates for removal in order to simplify the axis that we compare recipes

cvec = CountVectorizer(strip_accents=ascii)
cvecdata = cvec.fit_transform(recipes[0])

cvec_dense  = pd.DataFrame(cvecdata.todense(),
             columns=cvec.get_feature_names())

word_count = cvec_dense.sum(axis=0)    
cw = word_count.sort_values(ascending = False)
cw_dict = dict(cw)

# quick function to manually evaluate words that ought to be removed
# https://stackoverflow.com/questions/5844672/delete-an-item-from-a-dictionary

def removekey(d, key):
    r = dict(d)
    del r[key]
    return r

def eval_words(word_list):
    keeps = []
    nondescript = []
    
    nondescript_words = [] 
    keep_words = []
    
    for key,value in word_list.items():
            word_eval = input('Keep {}: {}, y or n?'.format(key,value))
        
            if word_eval == 'n':
                nondescript_words.append(key)
            else:
                keep_words.append(key)
            
            remaining_list = removekey(word_list,key)
            
            if len(nondescript_words) % 100 == 0:
                nondescript = nondescript + nondescript_words
                keeps = keeps + keep_words
                
                # empty holding lists
                keep_words = []
                nondescript_words = []
                
                prompt_continue = input('Continue: yes or no?')
                if prompt_continue == "yes":
                    pass
                else:
                    # export lists as pickles for recovery
                    # store outside the environment to limit reprocessing
                    words_lists = (keeps,nondescript,remaining_list)
                    names = ("keeps","nondescript","remaining")
                    for index,word in enumerate(words_lists):
                        with open("assets/"+names[index]+".pickle","wb") as file:
                            pickle.dump(word,file)
                    return keeps, nondescript,remaining_list
    
    # export word lists for recovery
    # so we don't have to do this multiple times
    words_lists = (keeps,nondescript,remaining_list)
    names = ("keeps","nondescript","remaining")
    for index,word in enumerate(words_lists):
        with open("assets/"+names[index]+".pickle","wb") as file:
            pickle.dump(word,file)
    
    return keep_words, nondescript_words, remaining_list

keep, nondescript, remain = eval_words(cw)

# read in pickled results
keep_list = []
drop_list = []

names = ("_keeps","_nondescript")
for name in names:
    for index in range(6):
        with open("assets/"+str(index)+name+".pickle",'rb') as file_handle:
            if name == "_keeps":
                keep_list = keep_list + pickle.load(file_handle)
            elif name == "_nondescript":
                drop_list = drop_list + pickle.load(file_handle)

# fix human error from ingredient classifications

misclassed_words = ['dice','block','dipping','stems','liter','pestle','2lb','pad','addition','paleo',
                    'smaller','teaspoons','gf','meatles','anytime','xe4utet','almond','scallions',
                    'evoo','wing','non','meal','gala','escarole','nectarine','stuffing','ganache',
                    'speck','hefe','champignon','silver','blade','kabocha','goudak','lindt','quorn',
                    'choi','evoki','aioli','broil','drumette','tex','massamon','pao','steamer','dandelion',
                    'bonnet','rapini','cakes','yucatero','cheek','latin','jimmy','quahog','cone','durum',
                    'cornichons','banh','fryers','quantity','5tbsp','llime','chopping','spam','ink','plant',
                    'triangular','valencia','tubetti','tubettini','cavatelli','perhap','livers','bee',
                    'tartine','teacup','barlett','maker','xlour','jell','fat','free'
                   ]

for word in misclassed_words:
    if word in keep_list and word not in drop_list:
        drop_list.append(word)
        keep_list.remove(word)
        print('{} added to drop_list'.format(word))
    elif word in drop_list and word not in keep_list:
        keep_list.append(word)
        drop_list.remove(word)
        print('{} added to keep list'.format(word))
    elif word in drop_list and word in keep_list:
        print('! {} found on both lists !')
    elif word not in drop_list and word not in keep_list:
        print('! {} not found on either list ! You misspelled target word'.format(word))
    else:
        print('! Bigger problems !')

# at the conclusion of this, we have a list of words in ingredients that should be dropped in the text processing
# stored as a drop_list

# Pantry Items
# re-process ingredient list, this time removing the non-descript words identified above
# getting data ready for recommender

num_cores = multiprocessing.cpu_count()
inputs = short_recipes['ingredientLines']

if __name__ == "__main__":
    recipes_drops = Parallel(n_jobs=num_cores)(delayed(multi_process_ingredients)(i,join=0,
                                                                                  nondescript=1,
                                                                                  drop_words=drop_list
                                                                                 )
                                                                            for i in inputs)

pantry = ['oregano','garlic powder','ground cumin','onion powder','ground mustard','hot hungarian paprika',
          'mexican oregano','smoked paprika','dill weed','ground turmeric','ground ginger','ground cloves',
         'cumin seed','cayenne pepper','chili powder','ground thyme','celery seed','curry powder',
          'ground white pepper','paprika','ground nutmeg','old bay','maple syrup','thyme leaves',
          'ground black pepper','black pepper','black peppercorns','crushed red pepper flakes','whole oregano',
         'minced onion','fennel seed','cinnamon','dried basil','anise seed','bay leaves','bay leaf',
          'ancho chili powder','ground cloves','coriander','vanilla extract','italian seasoning',
          'apple cider vinegar','honey','corn starch','balsamic vinegar','bread crumbs','white wine vinegar',
         'soy sauce','ketchup','tomato ketchup','red wine vinegar','vegatable oil','canola oil','sherry',
          'baking powder','baking soda','molasses','peanut butter','olive oil','extra virgin olive oil','salt',
          'sea salt','kosher salt','white vinegar','egg','eggs','egg whites','egg yolk','sugar','flour','evoo',
          'butter']
    
# this eliminates ingredients that have been wholly reduced to blanks
test = [[[ingredient for 
                   ingredient in recipe if ingredient != ''] 
                  for recipe in item] 
                 for item in recipes_drops]

# we're going to take all the ingredients from every recipe and string them together
# then take the set of that to find every unique ingredient
ingredient_master = []
for items in test:
    for recipe in items:
        ingredient_master = list(set(ingredient_master + recipe))

# from the ingredient_master list, we eliminate ingredients that bear substantial similiarity to our pantry items
# because these ingredients are IN our pantry, they are not essential for determining overall recipe similiarity
# in fact, it gives us more degrees of freedom to find a match by increasing the range of possible flavor profiles
# of a related recipe -- because we go into the pantry and pull out different spices other than those in our target
# recipe
# this will fuzzy matching (set token matching)

def match_pantry(ingredient_list,pantry_items=pantry):
    pantry_matches = []

    for ingredient in ingredient_master:
        if ingredient in pantry_items:
            pantry_matches.append(ingredient)
        else:
            for item in pantry_items:
                if fuzz.ratio(item,ingredient) > 70:
                    pantry_matches.append(ingredient)
    return pantry_matches

num_cores = multiprocessing.cpu_count()
inputs = ingredient_master

if __name__ == "__main__":
    pantry_matches = Parallel(n_jobs=num_cores)(delayed(match_pantry)(i) for i in inputs)

# this returns our list of ingredients that we can pull out of our pantry
# we should incorporate this into our ingredient processing and remove those items from the recipes to reduce complexity
# pantry_matches