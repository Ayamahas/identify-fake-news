#import the libraries i use it in my project
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import  recall_score,precision_score,accuracy_score
from sklearn.pipeline import make_pipeline
# Read the true and fake news datasets
true_news = pd.read_csv("True.csv")
fake_news = pd.read_csv("Fake.csv")
print("Data preproceccing of : ")
print("The data name (Ture.csv) :\n",true_news.isnull().sum(),"\n\n")
print("The data name (Fake.csv) :\n",fake_news.isnull().sum())
''' df1 = true_news.fillna(' ')
    df2 = fake_news.fillna(' ') 
After this step we no longer have any missing datapoints, you can check that using the isnull().sum()
our data haven't any missing value so i  didn't use df1,df2 '''
#add a new cloumn to classify the news  (true news =1,fake news=0)
true_news["target"] = 1
fake_news["target"] = 0
#my dataset is two part so will connect it with each other randomly (frac=1 means that we take 100% of rows original dataset)
all_news_data = pd.concat([true_news, fake_news], ignore_index=True).sample(frac=1)
# Split the new data for feature and label to use ro train model
feature = all_news_data['text']
label = all_news_data['target']
# Split the data for training and test sets the size of test set is 30% from alldata set
X_train, X_test, y_train, y_test = train_test_split(feature, label, test_size=0.3)
# Create a pipeline with TF-IDF vectorizer and Multinomial Naive Bayes classifier
model_nb = make_pipeline(TfidfVectorizer(), MultinomialNB())
model_nb.fit(X_train, y_train)
# test this model
predictions_nb = model_nb.predict(X_test)
# Create a pipeline with TF-IDF vectorizer and Logistic Regression classifier
model_lr = make_pipeline(TfidfVectorizer(), LogisticRegression())
model_lr.fit(X_train, y_train)
# test this model
predictions_lr = model_lr.predict(X_test)


# Print the evaluation for both models
print("Naive Bayes Model:")
print("accuracy_nb = ",accuracy_score(y_test,predictions_nb))
print("precision_nb = ",precision_score(y_test,predictions_nb))
print("recall_nb = ",recall_score(y_test,predictions_nb))
print("Logistic Regression Model:")
print("accuracy_lr = ",accuracy_score(y_test,predictions_lr))
print("precision_lr = ",precision_score(y_test,predictions_lr))
print("recall_lr = ",recall_score(y_test,predictions_lr))
from sklearn.svm import SVC
model_svm = make_pipeline(TfidfVectorizer(), SVC())
model_svm.fit(X_train, y_train)
predictions_svm = model_svm.predict(X_test)
print("SVM Model : ")
print("accuracy_svm = ",accuracy_score(y_test,predictions_svm))
print("precision_svm = ",precision_score(y_test,predictions_svm))
print("recall_svm = ",recall_score(y_test,predictions_svm))
from sklearn.ensemble import RandomForestClassifier
model_rf = make_pipeline(TfidfVectorizer(), RandomForestClassifier())
model_rf.fit(X_train, y_train)
predictions_rf = model_rf.predict(X_test)
print("Random Forest Model:")
print("accuracy_rf =", accuracy_score(y_test, predictions_rf))
print("precision_rf =", precision_score(y_test, predictions_rf))
print("recall_rf =", recall_score(y_test, predictions_rf))