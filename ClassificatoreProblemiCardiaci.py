
import sys

from SPARQLWrapper import SPARQLWrapper, JSON
from sklearn.model_selection import GridSearchCV , StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from scipy.stats import hypergeom
import bnlearn as bn
from sklearn.model_selection import train_test_split , cross_val_score
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn import svm
from sklearn.svm import SVC
from numpy import arange
from sklearn.naive_bayes import GaussianNB




endpoint_url = "https://query.wikidata.org/sparql"

query = """SELECT ?numLabel ?num1Label ?ageAdultLabel
WHERE 
{
   wd:Q41861 wdt:P5135 ?num.
   wd:Q762713 wdt:P5135 ?num1.
   wd:Q1202615 wdt:P5136 ?ageAdult

  SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }
}"""
def get_results(endpoint_url, query):
    user_agent = "WDQS-example Python/%s.%s" % (sys.version_info[0], sys.version_info[1])
    # TODO adjust user agent; see https://w.wiki/CX6
    sparql = SPARQLWrapper(endpoint_url, agent=user_agent)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    return sparql.query().convert()

one = OneHotEncoder(sparse=False)

results = get_results(endpoint_url, query)

for result in results["results"]["bindings"]:
    hypertension = int(result["numLabel"]["value"])
    hypercholesterol = int(result["num1Label"]["value"])
    adult = int(result["ageAdultLabel"]["value"])

data = pd.read_csv("pyRDF2Vec/samples/countries-cities/heart.csv")
data1=data
data.head()

data1["Sex"].values[data["Sex"].values == "M"] = 0
data1["Sex"].values[data["Sex"].values == "F"] = 1

data1["ExerciseAngina"].values[data["ExerciseAngina"].values == "N"] = 0
data1["ExerciseAngina"].values[data["ExerciseAngina"].values == "Y"] = 1


data1["ST_SlopeUp"] = 0
data1["ST_SlopeFlat"] = 0
data1["ST_SlopeDown"] = 0

data1["ST_SlopeUp"].values[data["ST_Slope"].values == "Up"] = 1
data1["ST_SlopeFlat"].values[data["ST_Slope"].values == "Flat"] = 1
data1["ST_SlopeDown"].values[data["ST_Slope"].values == "Down"] = 1

data1["ChestPainTypeATA"] = 0
data1["ChestPainTypeNAP"] = 0
data1["ChestPainTypeASY"] = 0
data1["ChestPainTypeTA"] = 0

data1["ChestPainTypeTA"].values[data["ChestPainType"].values == "TA"] = 1
data1["ChestPainTypeASY"].values[data["ChestPainType"].values == "ASY"] = 1
data1["ChestPainTypeNAP"].values[data["ChestPainType"].values == "NAP"] = 1
data1["ChestPainTypeATA"].values[data["ChestPainType"].values == "ATA"] = 1

data1["ResrtingECGNormal"] = 0
data1["RestingECGST"] = 0
data1["ResrtingECGLHV"] = 0

data1["ResrtingECGNormal"].values[data["RestingECG"].values == "Normal"] = 1
data1["RestingECGST"].values[data["RestingECG"].values == "ST"] = 1
data1["ResrtingECGLHV"].values[data["RestingECG"].values == "LHV"] = 1

data1 = data1.drop(columns=["RestingECG" , "ChestPainType", "ST_Slope"])


data["Hypertension"] = data['RestingBP']
data['Hypertension'].values[data['RestingBP'].values > hypertension] = 1
data['Hypertension'].values[data['RestingBP'].values <= hypertension] = 0

data["Hypercholesterol"] = data["Cholesterol"]
data['Hypercholesterol'].values[data["Cholesterol"].values > hypercholesterol] = 1
data['Hypercholesterol'].values[data["Cholesterol"].values <= hypercholesterol] = 0

data["LifePeriod"] = data["Age"]
data["LifePeriod"].values[data["Age"].values >= adult] = 1
data["LifePeriod"].values[data["Age"].values < adult] = 0

data["Ischemy"] = data["Oldpeak"]
data["Ischemy"].values[data["Oldpeak"].values > 2] = 2
data["Ischemy"].values[data["Oldpeak"].values < 2] = 1

data = data.drop(columns=["Age" , "Oldpeak", "MaxHR", "RestingBP", "Cholesterol" ])

data.head(100)

train, test = train_test_split(data , test_size=0.2, random_state=4)

X = data1.drop(columns=["HeartDisease"])
y = data1["HeartDisease"]
X_train , X_test , y_train , y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)


X.columns
X.head()



"""DAG = bn.structure_learning.fit(train, methodtype='naivebayes', root_node='HeartDisease')

print(DAG['adjmat'])

DAG = bn.independence_test(DAG, train, prune=True)

G = bn.plot(DAG, interactive=True)

bn.print_CPD(DAG)


model_mle = bn.parameter_learning.fit(DAG, train, methodtype='maximumlikelihood')

bn.print_CPD(model_mle)

#predict per le variabili secondo modelli di apprnedimento di probabilità
sum = 0
for example in test.index:
    q1 = bn.inference.fit(model_mle, variables=['HeartDisease'], evidence={
        "Hypercholesterol":test["Hypercholesterol"][example],
        "Hypertension":test["Hypertension"][example],
        "ST_Slope":test["ST_Slope"][example],
        "ExerciseAngina":test["ExerciseAngina"][example],
        "RestingECG":test["RestingECG"][example],
        "FastingBS":test["FastingBS"][example],
        "ChestPainType":test["ChestPainType"][example],
        "Sex":test["Sex"][example],
        "LifePeriod":test["LifePeriod"][example]
        })
        
    if(test["HeartDisease"][example] == 0):
        sum = sum + pow(q1.df["p"][1] , 2)
    else:
        sum = sum + pow(q1.df["p"][0] , 2)
    print(sum)

probability = 1-(sum / len(test.index))
print(probability)"""

"""#knn prediction
knn = KNeighborsClassifier(n_neighbors=150)
knn.fit(X_train , y_train)
scoreKnn = cross_val_score(knn , X_train , y_train)
scoreKnn
predicted = knn.predict(X_test)

print(accuracy_score(predicted , y_test))


#SVM prediction
regr = svm.LinearSVC(C=0.01)
regr.fit(X_train , y_train)
regr.score(X_train,y_train)

y_pred = regr.predict(X_test)
print(classification_report(y_test, y_pred))

clf = svm.SVC(kernel='linear' , C = 1, random_state=42)
scores = cross_val_score(clf , X_train , y_train, cv=10)
scores"""

#nested cv inizializziamo i classificatori 
clf1 = LogisticRegression(multi_class='multinomial' , solver = 'newton-cg', random_state=1)

clf2 = KNeighborsClassifier( algorithm='ball_tree' , leaf_size=50)

clf3 = DecisionTreeClassifier(random_state=1)

clf4 = SVC(random_state=1)

clf5 = RandomForestClassifier(random_state=1)

clf6 = GaussianNB()


#building the pipelines
pipe1 = Pipeline([('std' , StandardScaler()), ('clf1' , clf1)])

pipe2 = Pipeline([('std' , StandardScaler()) , ('clf2' , clf2)])

pipe4 = Pipeline([('std' , StandardScaler()) ,('clf4' , clf4)])


#setting the parameters grid
param_grid1 = [{'clf1__penalty': ['l2'], 'clf1__C': np.power(10., np.arange(-4, 4))}]

param_grid2 = [{'clf2__n_neighbors': list(range(1,10)), 'clf2__p': [1, 2]}]

param_grid3 = [{'max_depth': list(range(1 , 10)) + [None], 'criterion': ['gin1', 'entropy']}]

param_grid4 = [{'clf4__kernel': ['rbf'], 'clf4__C': np.power(10. , np.arange(-4, 4)), 
                'clf4__gamma': np.power(10., np.arange(-5, 0))}, 
                {'clf4__kernel': ['linear'], 'clf4__C': np.power(10. , np.arange(-4, 4))}]

param_grid5 = [{'n_estimators': [10, 100, 500, 1000, 10000]}]

param_grid6 = [{}]

#settare GridsearchCV multiple 1 per ogni algoritmo
gridcvs = {}
innder_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

for pgrid, est, name in zip((param_grid1, param_grid2, param_grid3 , param_grid4, param_grid5 , param_grid6), 
                            (pipe1, pipe2, clf3, pipe4 , clf5 , clf6),
                            ('Softmax', 'KNN', 'DTree', 'SVM', 'RForest' , 'NaiveBayes')):
    print(pgrid , est , name)
    print('ok')
    gcv = GridSearchCV(estimator= est, param_grid=pgrid, scoring='accuracy', n_jobs=-1, 
                       cv=innder_cv, verbose=0, refit=True)
    gridcvs[name] = gcv

outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

"""for name, gs_est in sorted(gridcvs.items()):
    nested_score = cross_val_score(gs_est, 
                                   X=X_train, 
                                   y=y_train, 
                                   cv=outer_cv,
                                   n_jobs=-1)
    print('%s | outer ACC %.2f%% +/- %.2f' % 
          (name, nested_score.mean() * 100, nested_score.std() * 100))"""
b = outer_cv.split(X_train , y_train)

for i, (train_index1 , test_index1) in enumerate(b):
    len(test_index1)
clf1.get_params().keys()
pipe1.get_params().keys()

for name, gs_est in sorted(gridcvs.items()):
    print(name)
    print(50 * '-', '\n')
    print('Alghorithm:', name)
    print('      Inner loop:')

    outer_scores = []
    outher_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
    b = outher_cv.split(X_train , y_train)
    for (train_idx, valid_idx) in b:   
        gridcvs[name].fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
        print('\n       Best ACC %.2f%%' % (gridcvs[name].best_score_ * 100))
        print('    Best Parameters:', gridcvs[name].best_params_)

        outer_scores.append(gridcvs[name].best_estimator_.score(X_train.iloc[valid_idx] , y_train.iloc[valid_idx]))
        print('         ACC (on outher test fold %.2f%%)' % (outer_scores[-1]*100)) 
    print('\n   Outher Loop:')
    print('      ACC %.2f%% +/- %.2f' % (np.mean(outer_scores) * 100, np.std(outer_scores) * 100))


outher_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
Xdata = train.drop(columns ="HeartDisease")
ydata = train["HeartDisease"]
b = outher_cv.split(Xdata , ydata)

for(train_idx, valid_idx) in b:
    DAG = bn.structure_learning.fit(train.iloc[train_idx], methodtype='naivebayes', root_node='HeartDisease')
    model_mle = bn.parameter_learning.fit(DAG, train.iloc[train_idx], methodtype='maximumlikelihood')
    sum = 0
    for example in train.iloc[valid_idx]:
        print('ok')
        q1 = bn.inference.fit(model_mle, variables=['HeartDisease'], evidence={
            "Hypercholesterol":train["Hypercholesterol"][example],
            "Hypertension":train["Hypertension"][example],
            "ST_Slope":train["ST_Slope"][example],
            "ExerciseAngina":train["ExerciseAngina"][example],
            "RestingECG":train["RestingECG"][example],
            "FastingBS":train["FastingBS"][example],
            "ChestPainType":train["ChestPainType"][example],
            "Sex":train["Sex"][example],
            "LifePeriod":train["LifePeriod"][example]
            })
        if(train["HeartDisease"][example] == 0):
            sum = sum + pow(q1.df["p"][1] , 2)
        else:
            sum = sum + pow(q1.df["p"][0] , 2)
        print(sum)
    
    




