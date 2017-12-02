# Correlation with seaborn with better axis labels
import seaborn as sns
import pandas as pd 
import matplotlib.pyplot as plt
attrition_num = pd.read_csv('C:/Users/monik/Desktop/DA/final_project/excel_txt/vstupy/att_num_csv.csv')
attrition_num.head()
labels = ['employee number', 'age', 
        'business travel', 'monhtly income', 	'department', 	'distance from home',
        'education', 'education field', 'environment satisfation', 'gender', 
        'job involvement', 'job level',	'job role', 'job satisfaction',
        'marital status', 'number companies worked', 'overtime', 'percent salary hike',
        'performance rating', 'relationship satisfaction', 'stock option level',	
        'total working years', 'training times last year', 'work life balance', 
       'years at company', 'years in current role', 'years since last promo', 
       'years with current manager', 'attrition']
corr = attrition_num.corr()
plt.figure(figsize = (12, 12))
sns.heatmap(corr, 
            xticklabels=labels,
            yticklabels=labels)
sns.set(font_scale = 3)
plt.show()
