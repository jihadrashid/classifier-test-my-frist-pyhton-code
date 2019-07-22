from sklearn import tree
from sklearn import naive_bayes
from sklearn import svm
x = [[152,50,30], [163,61,42], [175,73,45], [142,35,25], [168,56,38], [180,83,45], [171,56,36], [165,60,41], [185,78,45],
     [161,59,37], [178,71,43], [140,31,24], [181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39], [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]
y = ['female', 'female', 'male', 'female', 'male', 'male', 'male', 'female', 'male', 'female', 'male', 'female', 'male',
     'male', 'female', 'female', 'male', 'male', 'female', 'female', 'female', 'male', 'male']

clfv = svm.SVC(gamma='scale')
clfv = clfv.fit(x,y)
prediction = clfv.predict([[192,89,51]])
print(prediction)


gnb = naive_bayes.GaussianNB()
gnb = gnb.fit(x,y)
prediction = gnb.predict([[192,89,51]])
print(prediction)

clf = tree.DecisionTreeClassifier()
clf = clf.fit(x,y)
prediction = clf.predict([[192,89,51]])
print(prediction)

