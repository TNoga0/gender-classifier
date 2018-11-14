import pandas
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import tree
import sys

#defining some constants and lists
NUMBER_OF_FEATURES_NOSE = 10
NUMBER_OF_FEATURES_EYEBROWS = 10
NUMBER_OF_FEATURES_EYES = 10
NUMBER_OF_FEATURES_LIPS = 10
NUMBER_OF_FEATURES_CHIN = 10
NUMBER_OF_TRAINING_EXAMPLES = 100

#vector is the csv filename of input column vector
vector = sys.argv[2]

#defining necessary lists, as a reminder:
    #learning data
nose_learning = []
eyebrows_learning = []
eyes_learning = []
lips_learning = []
chin_learning = []
labels_learning = []
    #test data
nose_test = []
eyebrows_test = []
eyes_test = []
lips_test = []
chin_test = []
labels_test = []

def read_input(data, vec):
    vectorread = pandas.read_csv(str(vec))
    colnames = vectorread.columns
    output = []
    length = vectorread.ix[0].__len__()
    for i in range(0,length):
        output.append(vectorread.ix[0][i])
    for count in range (0,length):
        if output[count] == 0:
            data.drop(colnames[count], axis = 1, inplace = True)

def split_men_women(data, labels, amount_men, amount_women, id_men, id_women):
    training = NUMBER_OF_TRAINING_EXAMPLES
    quantity = amount_men - (training/2) if amount_men < amount_women else amount_women - (training/2)

    test = []
    learn = []
    for i in range(0,training/2):
        test.append(id_men[i])
        test.append(id_women[i])

    for i in range(0,quantity-1):
        learn.append(id_men[i+(training/2)+1])
        learn.append(id_women[i+(training/2)+1])

    data_train = data.ix[test[0:len(test)]]
    labels_train = labels.ix[test[0:len(test)]]
    data_learn = data.ix[learn[0:len(learn)]]
    labels_learn = labels.ix[learn[0:len(learn)]]

    return data_learn, labels_learn, data_train, labels_train


def Amount_men_women(label):
    length = label.__len__()
    men = 0
    women = 0
    index_men = []
    index_women = []
    for count in range(0,length):
        if(label.ix[count] == 1):
            women = women+1
            index_women.append(count)
        else:
            men = men + 1
            index_men.append(count)
    return men,women,index_men,index_women

#loading the input dataset
genderread = pandas.read_csv(str(sys.argv[1]))
columnnames = genderread.columns #loading column names into a list, so it would be easier to split the dataset
#splitting labels and data
data = genderread.drop('CLASS', axis = 1) #droping the labels column
labels = genderread['CLASS'] #defining labels

#dropping the unnecessary columns and splitting the dataset to have equal amount of both classes' samples:
read_input(data, vector)

men_am,women_am, ind_men, ind_women = Amount_men_women(labels)


data_learning, labels_learning, data_test, labels_test = split_men_women(data, labels, men_am, women_am, ind_men, ind_women)

#DEFINE CLASSIFIER
classifier_all = tree.DecisionTreeRegressor()

#FITTING
classifier_all.fit(data_learning,labels_learning)
print 'Done fitting'

#VALIDATE
labels_predict_all = classifier_all.predict(data_test)


print '----------------'
print 'ALL DATA RESULTS'
print(confusion_matrix(labels_test, labels_predict_all))
print(classification_report(labels_test, labels_predict_all))











