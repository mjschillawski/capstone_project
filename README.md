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
| [00_EDA.ipynb](https://github.com/mjschillawski/capstone_project/blob/master/00_EDA.ipynb) | Recipe Recommender System |
| [recommender.py]() |  |
| [Presentation Deck](https://docs.google.com/presentation/d/1u8gIq1u46CyaZ49r7KACPhh2l_XackVf73mayR2G110/edit?usp=sharing) | Slide deck |
| [assets](https://github.com/mjschillawski/capstone_project/tree/master/assets) | Supporting materials, working files and temporary staging |
| [presentation]() | Slide deck images |
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

- The recommendation engine, using cosine similarity, will measure the angular distance between each ingredient vector to 

---

## Future Extensions






- 