# -*- coding:utf-8 -*-
#__author__: 'Baojian Jiang'
#__time__:'2019/2/26'

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import math

auto_prices = pd.read_csv('imports-85.data')
print(auto_prices.head(20))
