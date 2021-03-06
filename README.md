# capstone_project
Capstone Project for General Assembly: Data Science Immersive Program

Michael Schillawski

Data Science Immersive, General Assembly, 10 April 2018

---

## Table of Contents

- [Repository Contents](#repository-contents) - Description of this repository's contents
- [The Problem](#problem) - Motivation for this project
- [Technical Documentation](#technical-documentation) - Documentation on methods
---

## Repository Contents

| FILENAME |  DESCRIPTION |
|:---------:|:-----------:|
| [README](./README.md) | Project documentation |
| [LICENSE](./LICENSE) | License |
| [capstone.ipynb](https://github.com/mjschillawski/capstone_project/blob/master/capstone.ipynb) | Recipe Recommender System notebook |
| [recommender.py](https://github.com/mjschillawski/capstone_project/blob/master/recommender.py) | Stand-alone Recommender Function|
| [Recommender_Demonstration](https://github.com/mjschillawski/capstone_project/blob/master/Recommender_Demonstration.ipynb) | Recommender Demonstration Notebook|
| [EDA_and_ScratchPad.ipynb](https://github.com/mjschillawski/capstone_project/blob/master/00_EDA.ipynb) | Exploration and Draft Functions |
| [capstone_presentation.pdf](https://github.com/mjschillawski/capstone_project/blob/master/capstone_presentation.pdf) | Slide deck |
| [eval_words_log.txt](https://github.com/mjschillawski/capstone_project/blob/master/eval_words_log.txt) | Stop Word Evaluation Log|
| [assets](https://github.com/mjschillawski/capstone_project/tree/master/assets) | Supporting materials, working files, pickles, and temporary staging |
| [presentation](https://github.com/mjschillawski/capstone_project/tree/master/presentation) | Slide deck images |
| [.gitignore](./.gitignore) | gitignore file |

---
## The Problem: Combatting Individual-Level Food Waste

When I plan to cook, I select a recipe and plan my grocery shopping based on that. Once I prepare the recipe, I often find that the recipe yields leftover _raw_ ingredients. The packaged raw ingredients don't fit perfectly with what I have planned. But this opens a new problem: I don't have a use for the leftovers. So they rot.

I sought to build a recipe recommendation engine: based on the recipe that I intended to prepare, I wanted to identify other recipes that bore a substantial (but not exact) similarity (based on the ingredient list) to my intended meal, so that I could reuse the leftover raw ingredients in another application.

---

## Technical Documentation

### Data

I used a recipe corpus (a cookbook) from Luis Herranz (http://lherranz.org/datasets/), containing about 28,000 recipes acquired from Yummly. Apart from the metadat about the recipes, this was a convenient choice because each recipe's ingredients were stored as a list.

### Basics

While this began as a recommendation engine problem, it quickly devolved into a natural language processing one and a recursive one at that.

The data can be understood as follows:

- A cookbook (corpus) is a collection of recipes;

- A recipe (document) is a collection of ingredients;

- An ingredient (word) is the fundamental building block, and our unit of comparison.

### Algorithm

The essential structure of the algorithm is as follows:

- From our cookbook, distill each recipe down into just its ingredients.

- Each recipe's ingredients will need to be stripped down to its core, so that they can be compared:

  - Remove quantities;
  - Remove units and their abbreviations;
  - Create a custom stopword dictionary for culinary preparation words;
  - Reduce the dimensionality of the ingredients by removing ingredients commonly found in a kitchen; and
  - Further reduce the dimensionality of the ingredients by clustering similiar, substitutable ingredients.

- At this point, each recipe will be a vector of clusters, instead of a vector of ingredients.

- The recommendation engine, using cosine similarity, will measure the angular distance between each ingredient vector. Recipes that share more ingredients will have higher similarity scores. 

However, the eventual recommendation **should not be** the recipe(s) with the highest similarity. Here's the intuition: a recipe that shares 100% ingredient similarity with the target recipe is likely the same recipe. The nature of many recipe websites is to have many variations of the same recipe. While we have implemented some degree of freedom by reducing the dimensionality of the ingredients (subtracting common, baseline pantry ingredients and clustering ingredients), recipes with perfect similiarity are too similar to be a useful recommendation.

---

## Future Extensions

The kitchen is one of the last places in the home where data has yet to penetrate. At this point, bluetooth and wifi-connected kitchen appliances are novelties more than integrated solutions in the kitchen workflow.

This recipe recommender system, to help minimize home kitchen food waste, is the first thrust into this. Future development includes more robust pantry management. This early version relied on a static inventory of a pantry; future iterations will include dynamic pantry management, subtracting ingredients as they are consumed in prepared recipes.

This leads naturally into predictive grocery lists, advising the user when to replace pantry staples as they are consumed in cooking. From there, text-recognition for grocery receipts or integration into online grocery-ordering services would be able to provide real-time monitoring and inventory of the pantry. Further possibilities include integration into fitness and nutrition tracking apps, since, at that point, this reporting and recommendation suite would know what food options are available and when they are consumed.