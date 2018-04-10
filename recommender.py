# What's For Dinner?
# Authors: Michael Schillawski
# Data Science Immersive, General Assembly
# April 10, 2018
# Contact: mjschillawski@gmail.com

# Imports

import pickle

import pandas as pd
import numpy as np

recipe_sim = pd.read_csv('assets/recipe_sim.csv',index_col=0)

short_recipes = pd.read_csv('assets/short_recipes.csv',index_col=0)

# transform ingredient field back into list when importing from CSV
short_recipes['ingredientLines'] = short_recipes['ingredientLines'].apply(
    lambda x: [item for item in x.split('\'') if item not in ('\,','[',']',', ')])

# recommender

def recommender(recipe,max_thresh=0.9):
    ingredients = list(short_recipes[short_recipes['name']==recipe]['ingredientLines'])
    ingredients = [item for ingredient in ingredients for item in ingredient]
    
    match = recipe_sim[recipe].sort_values(ascending=False)[1:5]
    for m in match:
        if m <= max_thresh:
            best_match = match[match==m].index[0]
            break
    
    needs = list(short_recipes[short_recipes['name']==best_match]['ingredientLines'])
    needs = [item for need in needs for item in need]
    
    print('I recommend {}, with a similarity of {:3f}\n'.format(best_match,similarity))
    print('\nFor {}, you need:\n'.format(recipe))for ingredient in ingredients:
        print(ingredient)
    print('\nFor {}, you need:\n'.format(best_match))
    for need in needs:
        print(need)
    return None
