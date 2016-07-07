# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import sklearn.linear_model as sklm
import skflow as sf

def read_csv():
    train = pd.read_csv("~/PycharmProjects/soccer/train.csv", header=0)
    test = pd.read_csv("~/PycharmProjects/soccer/test.csv", header=0)
    cond_train = pd.read_csv("~/PycharmProjects/soccer/condition.csv", header=0)
    cond_test = pd.read_csv("~/PycharmProjects/soccer/condition_tests.csv", header=0)
    stadiums = pd.read_csv("~/PycharmProjects/soccer/stadium.csv", header=0)

    return train, test, stadiums, cond_train, cond_test

def make_train_test(train,test,cond_train,cond_test):

    train_test = train.append(test, ignore_index=True)
    cond_train_test = cond_train.append(cond_test, ignore_index=True)
    print(train_test)
    print(cond_train_test)
    #train_test_colnames = id,y,year,stage,match,gameday,time,home,away,stadium,tv
    #year
    years = pd.get_dummies(train_test["year"])
    #stage
    stage = pd.get_dummies(train_test["stage"])
    #match
    match = train_test["match"].str.split("節",expand=True)
    phase = pd.get_dummies(match[0])
    dayofphase = pd.get_dummies(match[0])
    #gameday
    gameday = train_test["gameday"].str.split("(", expand=True)
    date = gameday[0].str.split("/", expand=True)
    month = pd.get_dummies(date[0])
    day = pd.get_dummies(date[1])
    dayoftheweek = pd.get_dummies(gameday[1])
    #time
    time = train_test["time"].str.split(":", expand=True)
    hours = pd.get_dummies(time[0])
    minutes = pd.get_dummies(time[1])
    #home
    hometeam = pd.get_dummies(train_test["home"])
    #away
    awayteam = pd.get_dummies(train_test["away"])
    #stadium
    stadium = pd.get_dummies(train_test["stadium"])
    #tv
    tv = train_test["tv"].str.get_dummies("／")

#cond_train_test.col_names = id,home_score,away_score,weather,temperature,humidity,referee,home_team,home_01,home_02,home_03,home_04,home_05,home_06,home_07,home_08,home_09,home_10,home_11,away_team,away_01,away_02,away_03,away_04,away_05,away_06,away_07,away_08,away_09,away_10,away_11
    diff = cond_train_test["home_score"]-cond_train_test["away_score"]
    win = pd.DataFrame([1 if diff[i] > 0 else 0 for i in range(0,len(diff))])
    draw = pd.DataFrame([1 if diff[i] == 0 else 0 for i in range(0,len(diff))])
    lose = pd.DataFrame([1 if diff[i] < 0 else 0 for i in range(0,len(diff))])

    weather = pd.get_dummies(cond_train_test["weather"])
    #train_test["id"],train_test["y"],years,stage,phase,dayofphase,month,day,dayoftheweek,hours,minutes,hometeam,awayteam,stadium,tv,cond_train_test["home_score"],cond_train_test["away_score"],win,draw,lose,weather
    train_test = pd.concat([train_test["id"],years,stage,phase,dayofphase,month,day,dayoftheweek,hours,minutes,hometeam,awayteam,stadium,tv,cond_train_test["home_score"],cond_train_test["away_score"],win,draw,lose,weather],axis=1)
    train_test = pd.DataFrame(train_test)

    Train = train_test[0:1721]
    Test = train_test[1721:2034]

    return Train, Test


if __name__ == '__main__':

    train, test, stadiums, cond_train, cond_test = read_csv()
    Train_Y = pd.DataFrame(train["y"])
    train = train.drop(["y"],axis=1)
    Train, Test = make_train_test(train,test,cond_train,cond_test)

    Train_X = pd.DataFrame(Train.drop(["id"],axis=1))
    Test_X = pd.DataFrame(Test.drop(["id"],axis=1))

    print(Train_X)
    print(Test_X)

    lcv = sklm.RidgeCV(alphas=(2.9725,2.9726,2.9727,2.9728,2.9729,2.9730,2.9731,2.9732,2.9733,2.9734,2.9735), cv=15, fit_intercept=False)
    lcvf = lcv.fit(Train_X, Train_Y)

    Lasso_model = sklm.Ridge(alpha=lcvf.alpha_)
    results = Lasso_model.fit(Train_X, Train_Y)

    print(lcvf.alpha_)
    print(results.coef_)
    Forecast = pd.DataFrame(results.predict(Test_X))

    submit = pd.concat([Test["id"],Forecast], ignore_index= True ,axis=1)
    submit = pd.DataFrame(submit)

    submit.to_csv("submit.csv")



    Reg = sf.TensorFlowDNNRegressor(hidden_units=[2000,4000,10000,4000,2000], n_classes=1)
    s = np.array(Train_Y.values)
    X_train = np.array(Train_X)
    X_test = np.array(Test_X)

    Reg.fit(X_train, s)
    forecast = Reg.predict(X_test)

    forecast = pd.DataFrame()

    submit1 = pd.concat([Test["id"],Forecast], ignore_index= True ,axis=1)
    submit1 = pd.DataFrame(submit1)
    submit1.to_csv("submit1.csv")
