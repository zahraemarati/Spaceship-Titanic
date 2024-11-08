import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings("ignore")
sns.set(style="darkgrid",font_scale=1.5)
pd.set_option("display.max.rows",None)
pd.set_option("display.max.columns",None)

from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, StackingClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler


train_df = pd.read_csv('/kaggle/input/spaceship-titanic/train.csv')
test_df = pd.read_csv('test.csv')
# test_df['Transported'] = np.nan

train_df.describe()
train_df.info()

# feature engineering on PassengerId

def passengerid_new_features(df):
    
    #Splitting Group and Member values from "PassengerId" column.
    df["Group"] = df["PassengerId"].apply(lambda x: x.split("_")[0])
    df["Member"] =df["PassengerId"].apply(lambda x: x.split("_")[1])
    
    #Grouping the "Group" feature with respect to "member" feature to check which group is travelling with how many members
    x = df.groupby("Group")["Member"].count().sort_values()
    
    #Creating a set of group values which are travelling with more than 1 members.
    y = set(x[x>1].index)
    
    #Creating a new feature "Solo" which will indicate whether the person is travelling solo or not.
    df["Travelling_Solo"] = df["Group"].apply(lambda x: x not in y)
    
    #Creating a new feature "Group_size" which will indicate each group number of members.
    df["Group_Size"]=0
    for i in x.items():
        df.loc[df["Group"]==i[0],"Group_Size"]=i[1]

passengerid_new_features(train_df)
passengerid_new_features(test_df)

train_df.drop(columns=["Group","Member"],inplace=True)
test_df.drop(columns=["Group","Member"],inplace=True)

def cabin_new_feature(df):
    df["Cabin"].fillna("np.nan/np.nan/np.nan",inplace=True)  #In this way we can split NaN values into all three categories
    
    df["Cabin_Deck"] = df["Cabin"].apply(lambda x: x.split("/")[0])
    df["Cabin_Number"]  = df["Cabin"].apply(lambda x: x.split("/")[1])
    df["Cabin_Side"] = df["Cabin"].apply(lambda x: x.split("/")[2])
    
    #Replacing string nan values to numpy nan values..
    cols = ["Cabin_Deck","Cabin_Number","Cabin_Side"]
    df[cols]=df[cols].replace("np.nan",np.nan)
    
    #Filling Missing Values in new features created.
    df["Cabin_Deck"].fillna(df["Cabin_Deck"].mode()[0],inplace=True)
    df["Cabin_Side"].fillna(df["Cabin_Side"].mode()[0],inplace=True)
    df["Cabin_Number"].fillna(df["Cabin_Number"].median(),inplace=True)

cabin_new_feature(train_df)
cabin_new_feature(test_df)

train_df["Cabin_Number"]=train_df["Cabin_Number"].astype(int)
test_df["Cabin_Number"]=test_df["Cabin_Number"].astype(int)

plt.figure(figsize=(15,5))
sns.histplot(x="Cabin_Number",data=train_df,hue="Transported",palette="Set2")
plt.title("Cabin_Number Distribution")
plt.xticks(list(range(0,1900,300)))
plt.vlines(300,ymin=0,ymax=550,color="black")
plt.vlines(600,ymin=0,ymax=550,color="black")
plt.vlines(900,ymin=0,ymax=550,color="black")
plt.vlines(1200,ymin=0,ymax=550,color="black")
plt.vlines(1500,ymin=0,ymax=550,color="black")
plt.show()

#  Creating New Feature "Cabin_Regions" From "Cabin_Number"
def cabin_regions(df):
    df["Cabin_Region1"] = (df["Cabin_Number"]<300)
    df["Cabin_Region2"] = (df["Cabin_Number"]>=300) & (df["Cabin_Number"]<600)
    df["Cabin_Region3"] = (df["Cabin_Number"]>=600) & (df["Cabin_Number"]<900)
    df["Cabin_Region4"] = (df["Cabin_Number"]>=900) & (df["Cabin_Number"]<1200)
    df["Cabin_Region5"] = (df["Cabin_Number"]>=1200) & (df["Cabin_Number"]<1500)
    df["Cabin_Region6"] = (df["Cabin_Number"]>=1500)

train_df['Cabin_Number'] = train_df['Cabin_Number'].astype('int')
test_df["Cabin_Number"]=test_df["Cabin_Number"].astype(int)

cabin_regions(train_df)
cabin_regions(test_df)

train_df.drop(columns=["Cabin_Number"],inplace=True)
test_df.drop(columns=["Cabin_Number"],inplace=True)

def age_group(df):
    age_group  = []
    for i in df["Age"]:
        if i<=12:
            age_group.append("Age_0-12")
        elif (i>12 and i<=18):
            age_group.append("Age_13-18")
        elif (i>18 and i<=25):
            age_group.append("Age_19-25")
        elif (i>25 and i<=32):
            age_group.append("Age_26-32")
        elif (i>32 and i<=50):
            age_group.append("Age_33_50")
        elif (i>50):
            age_group.append("age_50+")
        else:
            age_group.append(np.nan)
        
    df["Age Group"] = age_group

train_df['Age'].fillna(train_df['Age'].mean(), inplace=True)
test_df['Age'].fillna(train_df['Age'].mean(), inplace=True)

age_group(train_df)
age_group(test_df)

# Creating New Features Using All Expenditude Features
exp_cols = ["RoomService","FoodCourt","ShoppingMall","Spa","VRDeck"]
train_df['Total Expenditure'] = train_df[exp_cols].sum(axis=1)
test_df['Total Expenditure'] = test_df[exp_cols].sum(axis=1)

def expenditure_category(df):
    expense_category = []
    
    for i in df["Total Expenditure"]:

        if i==0:
            expense_category.append("No Expense")
        elif (i>0 and i<=716):
            expense_category.append("Low Expense")
        elif (i>716 and i<=1441):
            expense_category.append("Medium Expense")
        elif (i>1441):
            expense_category.append("High Expense")
    
    df["Expenditure Category"] = expense_category

expenditure_category(train_df)
expenditure_category(test_df)

# Handling Missing Values
category_cols_train = train_df.select_dtypes(include=["object","bool"]).columns.tolist()
num_cols_train = train_df.select_dtypes(include=["int64","float64"]).columns.tolist()

category_cols_test = test_df.select_dtypes(include=["object","bool"]).columns.tolist()
num_cols_test = test_df.select_dtypes(include=["int64","float64"]).columns.tolist()

imputer1 = SimpleImputer(strategy="most_frequent")
imputer2 =  KNNImputer(n_neighbors=3)

def fill_missingno_train(df):
    df[category_cols_train] = imputer1.fit_transform(df[category_cols_train])
    df[num_cols_train] = imputer2.fit_transform(df[num_cols_train])

def fill_missingno_test(df):
    df[category_cols_test] = imputer1.fit_transform(df[category_cols_test])
    df[num_cols_test] = imputer2.fit_transform(df[num_cols_test])

fill_missingno_train(train_df)
fill_missingno_test(test_df)

passId_df = test_df[["PassengerId"]]
cols = ["PassengerId","Cabin","Name"]
train_df.drop(columns =cols, inplace=True)
test_df.drop(columns=cols, inplace=True)

# Applying Log Transformation on Expenditure Features
cols = ['RoomService','FoodCourt','ShoppingMall','Spa','VRDeck','Total Expenditure']

for value in cols:
    train_df[value] = np.log(1+train_df[value])


x=1
plt.figure(figsize=(20,35))
for i in cols:
    plt.subplot(6,2,x)
    sns.distplot(train_df[i],color="green")
    plt.ylim(0,0.2)
    plt.title(f"{i} Distribution")
    plt.tight_layout()
    x+=1

# Changing Data-Type to Boolean
cols = ["CryoSleep","VIP","Cabin_Region1","Cabin_Region2","Cabin_Region3", "Travelling_Solo","Cabin_Region4",
       "Cabin_Region5","Cabin_Region6"]

train_df[cols] = train_df[cols].astype(bool)
test_df[cols] = test_df[cols].astype(bool)

# Feature Encoding
nominal_cat_cols = ["HomePlanet","Destination"]
ordinal_cat_cols = ["CryoSleep","VIP","Travelling_Solo","Cabin_Deck","Cabin_Side","Cabin_Region1","Cabin_Region2",
                    "Cabin_Region3","Cabin_Region4","Cabin_Region5","Cabin_Region6","Age Group",
                    "Expenditure Category"]

L_enc = LabelEncoder()
onehot_enc = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

train_df[ordinal_cat_cols] = train_df[ordinal_cat_cols].apply(L_enc.fit_transform)
test_df[ordinal_cat_cols] = test_df[ordinal_cat_cols].apply(L_enc.fit_transform)

one_hot_encoded = onehot_enc.fit_transform(train_df[nominal_cat_cols])
train_df = pd.concat([train_df, pd.DataFrame(one_hot_encoded, columns=onehot_enc.get_feature_names_out(nominal_cat_cols))], axis=1)
train_df = train_df.drop(columns=["HomePlanet", "Destination"])

one_hot_encoded = onehot_enc.fit_transform(test_df[nominal_cat_cols])
test_df = pd.concat([test_df, pd.DataFrame(one_hot_encoded, columns=onehot_enc.get_feature_names_out(nominal_cat_cols))], axis=1)
test_df = test_df.drop(columns=["HomePlanet", "Destination"])


train_df["Transported"].replace({False:0,True:1},inplace=True)

X = train_df.drop(columns=["Transported"])
y = train_df[["Transported"]]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Splitting Data For Model Which Don't Need Scaled Data
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=0)

# Splitting Data For Model Which Need Scaled Data
X_train_Sc, X_test_Sc, y_train_Sc, y_test_Sc = train_test_split(X_scaled,y,test_size=0.2,random_state=0)


# Model Building For Scaled Data
training_score = []
testing_score = []

def model_prediction(model):
    model.fit(X_train_Sc,y_train_Sc)
    x_train_pred1 = model.predict(X_train_Sc)
    x_test_pred1 = model.predict(X_test_Sc)
    a = accuracy_score(y_train_Sc,x_train_pred1)*100
    b = accuracy_score(y_test_Sc,x_test_pred1)*100
    training_score.append(a)
    testing_score.append(b)
    
    print(f"Accuracy_Score of {model} model on Training Data is:",a)
    print(f"Accuracy_Score of {model} model on Testing Data is:",b)
    print("\n------------------------------------------------------------------------")
    print(f"Precision Score of {model} model is:",precision_score(y_test_Sc,x_test_pred1))
    print(f"Recall Score of {model} model is:",recall_score(y_test_Sc,x_test_pred1))
    print(f"F1 Score of {model} model is:",f1_score(y_test_Sc,x_test_pred1))
    print("\n------------------------------------------------------------------------")
    print(f"Confusion Matrix of {model} model is:")
    cm = confusion_matrix(y_test_Sc,x_test_pred1)
    plt.figure(figsize=(8,4))
    sns.heatmap(cm,annot=True,fmt="g",cmap="summer")
    plt.show()


# Logistic-Regression Model
model_prediction(LogisticRegression())


# KNeighborsClassifier Model
model_prediction(KNeighborsClassifier())


# Support-Vector-Classifier Model
model_prediction(SVC())


# Naive-Bayes Model
model_prediction(GaussianNB())


# Model Building For Un-Scaled Data
def model_prediction(model):
    model.fit(X_train,y_train)
    X_train_pred = model.predict(X_train)
    X_test_pred = model.predict(X_test)
    a = accuracy_score(y_train,X_train_pred)*100
    b = accuracy_score(y_test,X_test_pred)*100
    training_score.append(a)
    testing_score.append(b)
    
    print(f"Accuracy_Score of {model} model on Training Data is:",a)
    print(f"Accuracy_Score of {model} model on Testing Data is:",b)
    print("\n------------------------------------------------------------------------")
    print(f"Precision Score of {model} model is:",precision_score(y_test,X_test_pred))
    print(f"Recall Score of {model} model is:",recall_score(y_test,X_test_pred))
    print(f"F1 Score of {model} model is:",f1_score(y_test,X_test_pred))
    print("\n------------------------------------------------------------------------")
    print(f"Confusion Matrix of {model} model is:")
    cm = confusion_matrix(y_test,X_test_pred)
    plt.figure(figsize=(8,4))
    sns.heatmap(cm,annot=True,fmt="g",cmap="summer")
    plt.show()


# Decision-Tree-Classifier Model
model_prediction(DecisionTreeClassifier())


# Random-Forest-Classifier Model
model_prediction(RandomForestClassifier())


# Ada-Boost-Classifier Model
model_prediction(AdaBoostClassifier())


# Gradient-Boosting-Classifier Model
model_prediction(GradientBoostingClassifier())


#  LGMB Classifier Model
model_prediction(LGBMClassifier())


# XGBClassifier Model
model_prediction(XGBClassifier())


# Cat-Boost-Classifier Model
model_prediction(CatBoostClassifier(verbose=False))


models = ["Logistic Regression","KNN","SVM","Naive Bayes","Decision Tree","Random Forest","Ada Boost",
          "Gradient Boost","LGBM","XGBoost","CatBoost"]

df = pd.DataFrame({"Algorithms":models,
                   "Training Score":training_score,
                   "Testing Score":testing_score})


df.plot(x="Algorithms",y=["Training Score","Testing Score"], figsize=(16,6),kind="bar",
        title="Performance Visualization of Different Models",colormap="Set1")
plt.show()

testing_score
# Hyper-Parameter Tunning of LGBM Model
model1 = LGBMClassifier()

parameters1 = {"n_estimators":[100,300,500,600,650],
              "learning_rate":[0.01,0.02,0.03],
              "random_state":[0,42,48,50],
               "num_leaves":[16,17,18]}

grid_search1 = GridSearchCV(model1, parameters1, cv=5, n_jobs=-1)

grid_search1.fit(X_train,y_train.values.ravel())
best_parameters1 = grid_search1.best_params_

# Creating LGBM Model Using Best Parameters.
model1 = LGBMClassifier(**best_parameters1)
model1.fit(X_train,y_train)
X_test_pred1 = model1.predict(X_test)
accuracy_score(y_test,X_test_pred1)


# Hyper-Parameter Tunning of CatBoost Model
model2 = CatBoostClassifier(verbose=False)

parameters2 = {"learning_rate":[0.1,0.3,0.5,0.6,0.7],
              "random_state":[0,42,48,50],
               "depth":[8,9,10],
               "iterations":[35,40,50]}

grid_search2 = GridSearchCV(model2, parameters2, cv=5, n_jobs=-1)

grid_search2.fit(X_train,y_train)

grid_search2.best_score_

best_parameters2 = grid_search2.best_params_

# Creating Cat Boost Model Using Best Parameters
model2 = CatBoostClassifier(**best_parameters2,verbose=False)
model2.fit(X_train,y_train)
X_test_pred2 = model2.predict(X_test)
accuracy_score(y_test,X_test_pred2)


# Hyper-Parameter Tunning of XGBoost Model
model3 = XGBClassifier()

parameters3 = {"n_estimators":[50,100,150],
             "random_state":[0,42,50],
             "learning_rate":[0.1,0.3,0.5,1.0]}

grid_search3 = GridSearchCV(model3, parameters3 , cv=5, n_jobs=-1)

grid_search3.fit(X_train,y_train)

best_parameters3 = grid_search3.best_params_

# Creating XGBoost Model Using Best Parameters
model3 = XGBClassifier(**best_parameters3)
model3.fit(X_train,y_train)
X_test_pred3 = model3.predict(X_test)
accuracy_score(y_test,X_test_pred3)

# Hyper Parameter Tunning of RandomForest Model
model4 = RandomForestClassifier()

parameters4 = {'n_estimators': [100,300,500,550],
               'min_samples_split':[7,8,9],
               'max_depth': [10,11,12], 
               'min_samples_leaf':[4,5,6]}

grid_search4 = GridSearchCV(model4, parameters4, cv=5, n_jobs=-1)
grid_search4.fit(X_train,y_train.values.ravel())
best_parameters4 = grid_search4.best_params_

# Creating Random Forest Model Using Best Parameters
model4 = RandomForestClassifier(**best_parameters4)
model4.fit(X_train,y_train)
X_test_pred4 = model4.predict(X_test)
accuracy_score(y_test,X_test_pred4)


# Stacking Classifier Model
stacking_model = StackingClassifier(estimators=[('CAT Boost', model2),
                                                ("XGBoost", model3),
                                                ('RF', model4)])

stacking_model.fit(X_train, y_train)
x_train_pred5 = stacking_model.predict(X_train)
x_test_pred5 = stacking_model.predict(X_test)


# Predicting Test Data
lr = LogisticRegression()
lr.fit(X_train, y_train)

pred = lr.predict(test_df)
passId_df["Transported"] = pred
passId_df["Transported"].replace({1:True,0:False},inplace=True)
passId_df.to_csv("spaceship_prediction_project.csv",index=False)
