import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import plotly.offline as py
py.init_notebook_mode(connected=True)
from sklearn.ensemble import RandomForestClassifier

# Data import
attrition = pd.read_csv('C:/Users/monik/Desktop/DA/final_project/excel_txt/vstupy/att_csv.csv')
attrition.head()
attritionnum = pd.read_csv('C:/Users/monik/Desktop/DA/final_project/excel_txt/vstupy/att_num_csv.csv')
attritionnum.head()

# Dataframe definition
df = pd.DataFrame(data=attritionnum, columns = ['employee_number', 'age', 
        'business_travel_num', 'monhtly_income', 	'department_num', 	'distance_home',
        'education', 'edu_field_num', 'environment_satisfation', 'gender_num', 'job_involvement',
       'job_level',	'job_role_num', 'job_satisfaction', 'marital_status_num', 'num_comp_worked',
       'overtime_num', 'percent_salary_hike', 'performance_rating', 'relationship_satisfaction', 
       'stock_option_level',	'total_working_years', 'training_times_last_y', 'work_life_balance', 
       'years_at_company', 'years_in_current_role', 'years_since_last_promo', 'years_with_curr_manager',
       'attrition_num'])  
df.head()

# Generating a random # between 1 and 0, if # <= 0.75 the observations goes to the group train
df['is_train'] = np.random.uniform(0, 1, len(df)) <= .75
df.head()

# Delete employee number - not needed, could mess the method
del df['employee_number']
df.head()

# Create two new dataframes, one with the training rows, one with the test rows
train, test = df[df['is_train']==True], df[df['is_train']==False]

# Show the number of observations for the test and training dataframes
print('Number of observations in the training data:', len(train))
print('Number of observations in the test data:',len(test))

# Create a list of the feature column's names - without 29th column attrition
features = df.columns[:27]

# Defining the variable to be predicted - the target
x_train = train
y_train = train['attrition_num']
train.head()

# Defining test
x_test = test
y_test = test['attrition_num']

# Create a random forest Classifier
clf = RandomForestClassifier(n_estimators=25, n_jobs=2, random_state=0)

# Train the Classifier to take the training features and learn how they relate to the training y (attrition_num)
clf.fit(train[features], y_train)

# Apply the Classifier we trained to the test data
clf.predict(test[features])

# View the predicted probabilities of the first 10 observations
clf.predict_proba(test[features])[0:10]
preds = clf.predict(test[features])

# Create confusion matrix, WHICH REALLY IS CONFUSING AT FIRST
pd.crosstab(test['attrition_num'], preds, rownames=['Actual Attrition'], colnames=['Predicted Attrition'])

# View a list of the features and their importance scores
list(zip(train[features], clf.feature_importances_))

# The accuracy_score function computes the accuracy, either the fraction (default) or the count (normalize=False) of correct predictions.
score = accuracy_score(y_test, preds)
score_count = accuracy_score(y_test, preds, normalize=False)
print(score)
print(score_count)

# Scatter plot  
y = ['age', 'business_travel_num', 'monthly_income', 'department_num', 'distance_home',
     'education', 'edu_field_num', 'environment_satisfation', 'gender_num',
     'job_involvement', 'job_level', 'job_role_num', 'job_satisfaction',
    'marital_status_num', 'num_comp_worked', 'overtime_num', 'percent_salary_hike',
    'performance_rating', 'relationship_satisfaction', 'stock_option_level', 
    'total_working_years', 'training_times_last_y', 'work_life_balance', 
    'years_at_company', 'years_in_current_role',     'years_since_last_promo',
    'years_with_curr_manager']
x = clf.feature_importances_
plt.figure(figsize=(15, 10))
plt.scatter(x, y, c=x, vmin=0, vmax=0.10, s=400, alpha = 0.75, cmap='plasma')
plt.colorbar()
#plt.ylabel('Attributes')
plt.xlabel('Feature Importance')
plt.yticks([])
#plt.xticks(rotation=90)
plt.title('Random Forest Feature Importance')
labels = ['age', 'business travel', 'monthly income', 'department', 'distance from home',
 'education level', 'education field', 'environment satisfation', 'gender', 'job involvement',
  'job level', 'job role', 'job satisfaction', 'marital status', 'number companies worked',
   'overtime', 'percent salary hike', 'performance rating', 'relationship satisfaction', 
    'stock option level', 'total working years', 'training times last year', 
    'work life balance', 'years at company', 'years in current role', 'years since last promotion',
     'years with current manager']
for label, x, y in zip(labels, x, y):
    plt.annotate(
        label,
        xy=(x, y), xytext=(-20, 20),
        textcoords='offset points', ha='left', va='top',
        bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.5),
        arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))
plt.show() 
