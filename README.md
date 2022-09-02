# Earth Quake Damage Prediction
  
## Motivation

An earthquake is a calamitous occurrence that is detrimental to human interest and has an undesirable impact on the environment. Earthquakes have always caused in calculable damage to structures and properties and caused the deaths of millions of people throughout the world. In order to minimize the impact of such an event, several national, international and transnational organizations take various disaster detection and prevention measures. Time and quantity of the organization's resources are limiting factors, and organization managers face several difficulties when it comes to the distribution of the resources.Leveraging the power of machine learning is a viable option to predict the degree of damage that is done to buildings.

  
## Problem statement

Leveraging the power of machine learning is a viable option to predict the degree of damage that is done to buildings. It can help identify safe and unsafe buildings which helps to predict damage prone areas and thus avoiding death and injuries resulting from an earthquake, while simultaneously making rescue efforts efficient.This is done by classifying these structures on a damage grade scale based on various factors like its age, foundation, number of floors, material used and several other parameters. Then the number of families and the probable casualties ward-by-ward in a district are taken into account.This enables distribution of relief forces proportionately ward-wise and its prioritization based on the extent ofdamage.

## Project objectives

Earthquake are quite fatal and can cause quite a loss. Those that occur in the workplace can cause harm to employees, environment and damage to the equipment. Industrial related accidents, injuries and fatality data demonstrate that continued efforts and effective measures are necessary to reduce the number of industrial accidents, illnesses and and fatalities.This prediction can help identify safe and unsafe buildings which helps to predict damage prone areas and thus avoiding death and injuries resulting from an earthquake, while simultaneously making rescue efforts efficient.

## Implementation is in 7 phases

1. Importing required Modules.
2. Extracting the data from Kaggle.
3. Analyzing the data using Exploratory Data Analysis.
4. Removal of unwanted columns .
5. Splitting train and test data.
6. Implementation of different models.
7. Visulazing confusion matrix.


 **Dataset**

The dataset used in this project is downloaded from Kaggle[7]. It is a dataset containing csv files:


 **Building\_structure.csv**

| **Variable** | **Description** |
| --- | --- |
| building\_id | A unique ID that identifies every individual building |
| district\_id | District where the building is located |
| vdcmun\_id | Municipality where the building is located |
| ward\_id | Ward number in which the building is located |
| count\_floord\_pre\_eq | Number of floors that the building had before the earthquake |
| count\_floors\_post\_eq | Number of floors that the building had after the earthquake |
| age\_building | Age of building(in years) |
| plinth\_area\_sq\_ft | Plinth area of the building(in square feet) |
| height\_ft\_pre\_eq | Height of the building before the earthquake(in feet) |
| height\_ft\_post\_eq | Height of the earthquake after the earthquake(in feet) |
| land\_surface\_condition | Surface condition of the land in which the building is built |
| foundation\_type | Type of foundation used in the building |
| roof\_type | Type of roof used in the building |
| ground\_floor\_type | Type of construction used in other floors |
| other\_floor\_type | Type of construction used in other floors |
| postion | Postion of the building |
| plan\_configuration | Building plan configurationm |
| has\_superstructure\_adobe\_mud | Indicates if the superstructure of the building is made of Abode/Mud |
| has\_superstructure\_mud\_mortar\_stone | Indicates if the superstructure of the building is made of mud mortar |
| has\_superstructure\_cement\_mortar\_brick | Indicates if the superstructure of the building is made of Mud Mortar- Brick |
| has\_superstructure\_timber | Indicates if the superstructure of the building is made of Timber |
| has\_superstructure\_bamboo | Indicates if the superstructure of the building is made of Bamboo |
| has\_superstructure\_rc\_non\_engineered | Indicates if the superstructure of the building is made of RC(Non Engineered) |
| has\_superstructure\_rc\_engineered | Indicates if the superstructure of the building is made of RC(Engineered) |
| has\_superstructure\_other | Indicates if the superstructure of the building is made of any other material |
| condition\_post\_eq | Actual condition of the building after the earthquake |

## Data Variables Description:

| Variable  | Description |
| ------------- | ------------- |
| area_assesed  |Indicates the nature of the damage assessment in terms of the areas of the building that were assessed  |
| building_id  | A unique ID that identifies every individual building  |
| damage_grade  |Damage grade assigned to the building after assessment (Target Variable)  |
| district_id  | District where the building is located |
| has_geotechnical_risk  |Indicates if building has geotechnical risks  |
| has_geotechnical_risk_fault_crack  | Indicates if building has geotechnical risks related to fault cracking  |
| has_geotechnical_risk_flood  |Indicates if building has geotechnical risks related to flood |
| has_geotechnical_risk_land_settlement  | Indicates if building has geotechnical risks related to land settlement |
| has_geotechnical_risk_landslide |Indicates if building has geotechnical risks related to landslide  |
| has_geotechnical_risk_liquefaction |Indicates if building has geotechnical risks related to liquefaction |
| has_geotechnical_risk_other |Indicates if building has any other  geotechnical risks  |
| has_geotechnical_risk_rock_fall  |Indicates if building has geotechnical risks related to rock fall |
| has_repair_started  |Indicates if the repair work had started |
| vdcmun_id  | Municipality where the building is located |


  

**Exploratory Data Analysis**

Exploratory Data Analysis refers to the critical process of performing initial investigations on data to discover patterns, to spot anomalies. The main anomalies found in the data is missing values and independency of target variable on other variables.

**Missing Values**

Missing is filled by method fillna( ).

**Chi-Square test**

A  **chi-squared test** , also written as _ **χ** _ **2****  test**, is a [statistical hypothesis test](https://en.wikipedia.org/wiki/Statistical_hypothesis_testing) that is [valid](https://en.wikipedia.org/wiki/Validity_(statistics)) to perform when the test statistic is [chi-squared distributed](https://en.wikipedia.org/wiki/Chi-squared_distribution) under the [null hypothesis](https://en.wikipedia.org/wiki/Null_hypothesis), specifically [Pearson's chi-squared test](https://en.wikipedia.org/wiki/Pearson%27s_chi-squared_test) and variants thereof. Pearson's chi-squared test is used to determine whether there is a [statistically significant](https://en.wikipedia.org/wiki/Statistical_significance) difference between the expected [frequencies](https://en.wikipedia.org/wiki/Frequency_(statistics)) and the observed frequencies in one or more categories of a [contingency table](https://en.wikipedia.org/wiki/Contingency_table).

def ChiSquareTest(cat,res\_train):

for c in cat:

print(c)

tab = pd.crosstab(res\_train['damage\_grade'], res\_train[c])

stat, p, dof, expected = chi2\_contingency(tab)

print('dof=%d' % dof)

prob = 0.95

critical = chi2.ppf(prob, dof)

print('probability=%.3f, critical=%.3f, stat=%.3f' % (prob, critical, stat))

if abs(stat) \>= critical:

print('Dependent (reject H0)')

else:

print('Independent (fail to reject H0)')

# H0(target variable is independent on given attribute)

  **Removing Unwanted Columns**

After performing chi-square we get to know the unwanted columns by rejecting the H0 or failed to reject H0.

  **Splitting train and test data.**

We split the dataset into two different datasets they are test data and train data.

X\_train, X\_test,y\_train, y\_test = train\_test\_split(data, z\_train, test\_size=0.2)

- X\_train-the data used to train the model.
- y\_train-the train data of model.
- X\_test- the data used to test the model.
- y\_test-the test data of model.

# ALGORITHMS

 **Supervised Learning**

The dataset has been cleaned, pre-processed and analyzed for understanding the dataset. After such a process, and yet before coming to modeling, the dataset has to split up into two parts: Train and Test dataset. The training dataset is used to train the algorithm used to prepare an algorithm to comprehend. To learn and deliver results. It incorporates both input data and the desired output. The test datacollection is utilized to assess how well the algorithm was prepared with the traineddataset.

By using the Supervised learning Algorithms to train the dataset and also to test,predictions were made as to the desired outcome. The system was able to split, trainand test the dataset. Along with that, the feature importance was also given as theoutput where it had the percentage of these possibilities in occurring in the merefuture datasets that would be added.

The aim for us is to predict the Damage grade assigned to the building after assessment of Earth Quake.

The following are Supervised Algorithms which we used for prediction

- Decision Tree,
- Naïve Bayes'.
- Neural Networks.

**Naïve Bayes**

Naive Bayes is a classification algorithm for binary (two-class) and multi-class classification problems.

Bayes' Theorem finds the probability of an event occurring given the probability of another event that has already occurred. Bayes' theorem is stated mathematically as the following equation

p(cj | d ) = p(d | cj ) \*p(cj )/ p(d)

• p(cj | d) = probability of instance d being in class cj , This is what we are trying to compute

• p(d | cj ) = probability of generating instance d given class cj , We can imagine that being in class cj , causes you to have feature d with some probability

• p(cj ) = probability of occurrence of class cj , This is just how frequent the class cj , is in our database

• p(d) = probability of instance d occurring This can actually be ignored, since it is the same for all classes

Since our target variable has 5 different values our classification is multi-class classification.

naive\_bayes = GaussianNB()

naive\_bayes.fit(X\_train,y\_train)

prediction= naive\_bayes.predict(X\_test)

 **Random Forest**

Random Forest is a popular machine learning algorithm that belongs to the supervised learning technique. It can be used for both Classification and Regression problems in ML. It is based on the concept of ensemble learning, which is a process of combining multiple classifiers to solve a complex problem and to improve the performance of the model.

As the name suggests, "Random Forest is a classifier that contains a number of decision trees on various subsets of the given dataset and takes the average to improve the predictive accuracy of that dataset." Instead of relying on one decision tree, the random forest takes the prediction from each tree and based on the majority votes of predictions, and it predicts the final output.

random\_forest = RandomForestClassifier( criterion='entropy',n\_estimators=50,max\_features='log2', max\_depth=4,n\_jobs=-1, random\_state=0)

random\_forest.fit(X\_train,y\_train)

prediction= random\_forest.predict(X\_test)

**Neural Networks**

The Model is a Neural network with 307 inputs and 3 hidden layers and 8 categories/bins:

- The neural Network Architechture is created using torch nn module, which has 8 output. Log\_softmax is performed to get the label.
- nn.dropout(.02) is used so that the network does not over fit.
- nn.NLLLoss is used to calcuate the loss. since the output is a softmax which is in the form of exponential, Natural Log loss will be best to get the loss
- loss.backward() performs the backward pass
- optimizer.step() updates the new weight
- optimizer.zero\_grad() sets the grdient to zero for next backword propagation
- model.eval() freezes the gradient and removes the dropout for evaluation/validation during training

Parameter tuning for the model:

- Epochs
- Learning rate
- Mini Batch size
- Number of Hidden layers
- Number of Hidden nodes

 
 **Black BoxTesting**

When applied to machine learning models, black box testing would mean testing machine learning models without knowing the internal details such as features of the machine learning model, the algorithm used to create the model etc. The challenge, however, is to verify the test outcome against the expected values that are known beforehand.

| **Input** | **Actual Output** | **Predicted Output** |
| --- | --- | --- |
| [16,6,324,0,0,0,22,0,0,0,0,0,0] | 0 | 0 |
| [16,7,263,7,0,2,700,9,10,1153,832,9,2] | 1 | 1 |

**Example Black Box Testing**

The model gives out the correct output when different inputs are given which are mentioned in Table. Therefore the program is said to be executed as expected or correct program.

**Confusion Matrix**

A confusion matrix is a technique for summarizing the performance of a classification algorithm.Classification accuracy alone can be misleading if you have an unequal number of observations in each class or if you have more than two classes in your dataset.Calculating a confusion matrix can give you a better idea of what your classification model is getting right and what types of errors it is making.

Confusion matrix is sqaure matrix of size n where n is the number of values for targeted variable.

Let us consider a confusion matrix A nxn

A[i,j] indicates the number of times i is predicted as j

In our case n=5

0th row - grade1,1st row - grade 2,2nd row - grade 3,3rd row - grade 4,4th row - grade 5

![Confusion Matrix of Decision Tree](https://github.com/kalyan-karnati/Earth-Quake-Damage-prediction/blob/main/Pictures/dt.png)

![Confusion Matrix of Naïve Bayes](https://github.com/kalyan-karnati/Earth-Quake-Damage-prediction/blob/main/Pictures/nb.png)

![Confusion Matrix of Random Forest](https://github.com/kalyan-karnati/Earth-Quake-Damage-prediction/blob/main/Pictures/rf.png)


# Conclusion

##

Thus, the aim of this project is to predict the damage grade of buildings. The Random forest algorithm is high accurate in predicting compared to Naive bayes and Decision Tree.The use K-folds the accuracy of prediction is increasing the accuracy is directly proportion to number of folds.It is also seen that damage grade 1,5 have high accuracy in prediction.

**Accuracy comparision**

**Accuracy without folds**
![Accuracy without folds](https://github.com/kalyan-karnati/Earth-Quake-Damage-prediction/blob/main/Pictures/accuracy.png)

**Accuracy with folds**
![Accuracy with folds](https://github.com/kalyan-karnati/Earth-Quake-Damage-prediction/blob/main/Pictures/accuracyUsing%20Folds.png)

| **Algorithm** | **Accuracy** |
| --- | --- |
| Decision Tree | 66.19 |
| Naive Bayes | 67.01 |
| Neural Networks | 70.59 |



## Future scope
In future we plan to enable maps and show colors for different grades for different building.And also to host our application on web.Also, in future we would like to collect data from real time database and a dataset with more variation and a higher quality can really boost the accuracy of our current models. Also we think that using more complex models like artificial neural networks, or applying deep learning.

Maintaing a realtime data of building structure would help in better prediction and

Using hybrid models for increasing accuracy.

Due to time and knowledge constraints we could not develop great UI/UX. As an improvement to this model, we will give priority to use progressive web app that uses better rendering tools such as angularJS that improves client side UI/UX experience.

# REFERENCES

**[1]Long Wang, Xiaoqing Wang, Aixia Dou, Dongliang Wang**"Study on construction seismic damage loss assessment using RS and GIS" International Symposium on Electromagnetic compatibility, 2014.

**[2]****Roxane Mallouhy, Chady Abou Jaoude,Christophe Guyeux ,Abdallah Makhoul**,"Earthquake event prediction using various machine learning algorithms".

**[3]****G. Molchan and V. Keilis-Borok**,"Earthquake prediction: probabilistic aspect".

**[4]****H Takata, H. Nakamura, T Hachino**,"On prediction of electric power damage by typhoons in each district in Kagoshima Prefecture via LRM and NN".

**[5]****Ramli Adnan. Abd Manan Samad, Zainazlan Md Zain, Fazlina Ahmat Ruslan**,"5 hours flood prediction modeling using improved NNARX structure: case study Kuala Lumpur"

**[6]Dezhang Sun, Baitao Sun,**"Rapid prediction of earthquake damage to buildings based on fuzzy analysis"2010 Seventh International Conference on Fuzzy Systems and Knowledge Discovery, vol. 3, p. 1332-1335, 2010.

**[7][https://www.kaggle.com/arpitr07/predict-building-damage-grade/output](https://www.kaggle.com/arpitr07/predict-building-damage-grade/output)

