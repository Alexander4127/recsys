## Algorithm implementation

Each method is an inheritor of the base class `Model` and implements the following methods

- `fit` perfoms initial training of the model.
- `refit` further trains it using data from one iteration.
- `predict` gets the model forecast.
- `item_popularity` estimates popularity for each product.

Experiments in several formulations of the problem are presented in the file `bandits.ipynb`.
