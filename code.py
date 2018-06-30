from sklearn import tree

clf = tree.DecisionTreeClassifier()

# [freshman_gpa, sophomore_gpa, junior_gpa]
X = [[3.3, 3.8, 4.0], [2.8, 3.1, 2.7], [3.8, 2.1, 1.7], [3.2, 2.8, 3.3], [3.9, 2.9, 4.0],
     [1.9, 2.0, 2.8], [3.8, 3.6, 3.0],
     [2.8, 2.5, 2.1], [3.4, 2.0, 0], [2.8, 3.1, 2.7], [2.8, 3.1, 3.0]]
# [thou shall fail or pass(graduate)]
Y = ['Graduated', 'Failed', 'Failed', 'Failed', 'Graduated', 'Failed', 'Graduated', 'Failed',
     'Failed', 'Graduated', 'Graduated']


# train them on our data
clf = clf.fit(X, Y)

prediction = clf.predict([[3.6, 3.98, 3.7]])

# print your prediction
print(prediction)
