
import torch
import numpy as np
import pandas as pd
import re
import nltk
import warnings
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from scipy import sparse
import xgboost as xgb
from lightgbm import LGBMClassifier

from tqdm.notebook import tqdm

nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
warnings.filterwarnings('ignore')
