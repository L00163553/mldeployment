"""
# 
# File          : model.py
# Created       : 25/05/22 2:53 PM
# Author        : Ron Greego
# Version       : v1.0.0
# Description   :
#
"""

from pycaret.regression import setup, create_model, save_model
from pycaret.datasets import get_data

df = get_data("insurance")

r1 = setup(df, target='charges', session_id=123,
           normalize=True,
           polynomial_features=True, trigonometry_features=True,
           feature_interaction=True,
           bin_numeric_features=['age', 'bmi'])

# train a model
model = create_model('lr')

# save pipeline/model
save_model(model, model_name='deployment')
