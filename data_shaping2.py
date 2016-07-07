# -*- coding: utf-8 -*-
import pandas as pd
import sklearn.linear_model as sklm

def read_csv():
    data = pd.read_csv("~/PycharmProjects/soccer/data.csv", header=0)
    test = pd.read_csv("~/PycharmProjects/soccer/test.csv", header=0)
    stadiums = pd.read_csv("~/PycharmProjects/soccer/stadium.csv", header=0)
    cond_train = pd.read_csv("~/PycharmProjects/soccer/condition.csv", header=0)
    cond_test = pd.read_csv("~/PycharmProjects/soccer/condition_test.csv", header=0)
