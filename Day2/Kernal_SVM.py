import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
import numpy

train_df = pd.read_csv('credit_card_default_train.csv')
test_df = pd.read_csv('credit_card_default_test.csv')

# replacing balance limit v1
train_df['Balance_Limit_V1'] = train_df['Balance_Limit_V1'].replace('2.5M', 1)
train_df['Balance_Limit_V1'] = train_df['Balance_Limit_V1'].replace('1.5M', 0.6)
train_df['Balance_Limit_V1'] = train_df['Balance_Limit_V1'].replace('1M', 0.4)
train_df['Balance_Limit_V1'] = train_df['Balance_Limit_V1'].replace(' 500K', 0.2)
train_df['Balance_Limit_V1'] = train_df['Balance_Limit_V1'].replace('400K', 0.16)
train_df['Balance_Limit_V1'] = train_df['Balance_Limit_V1'].replace('300K', 0.12)
train_df['Balance_Limit_V1'] = train_df['Balance_Limit_V1'].replace('200K', 0.08)
train_df['Balance_Limit_V1'] = train_df['Balance_Limit_V1'].replace('100K', 0.04)
test_df['Balance_Limit_V1'] = test_df['Balance_Limit_V1'].replace('2.5M', 1)
test_df['Balance_Limit_V1'] = test_df['Balance_Limit_V1'].replace('1.5M', 0.6)
test_df['Balance_Limit_V1'] = test_df['Balance_Limit_V1'].replace('1M', 0.4)
test_df['Balance_Limit_V1'] = test_df['Balance_Limit_V1'].replace(' 500K', 0.2)
test_df['Balance_Limit_V1'] = test_df['Balance_Limit_V1'].replace('400K', 0.16)
test_df['Balance_Limit_V1'] = test_df['Balance_Limit_V1'].replace('300K', 0.12)
test_df['Balance_Limit_V1'] = test_df['Balance_Limit_V1'].replace('200K', 0.08)
test_df['Balance_Limit_V1'] = test_df['Balance_Limit_V1'].replace('100K', 0.04)

# replacing gender
train_df['Gender'] = train_df['Gender'].replace('M', 1)
train_df['Gender'] = train_df['Gender'].replace('F', -1)
test_df['Gender'] = test_df['Gender'].replace('M', 1)
test_df['Gender'] = test_df['Gender'].replace('F', -1)

# replacing educational status
train_df['EDUCATION_STATUS'] = train_df['EDUCATION_STATUS'].replace('Graduate', 1)
train_df['EDUCATION_STATUS'] = train_df['EDUCATION_STATUS'].replace('High School', 2)
train_df['EDUCATION_STATUS'] = train_df['EDUCATION_STATUS'].replace('Other', 3)
test_df['EDUCATION_STATUS'] = test_df['EDUCATION_STATUS'].replace('Graduate', 1)
test_df['EDUCATION_STATUS'] = test_df['EDUCATION_STATUS'].replace('High School', 2)
test_df['EDUCATION_STATUS'] = test_df['EDUCATION_STATUS'].replace('Other', 3)

# replacing marital status
train_df['MARITAL_STATUS'] = train_df['MARITAL_STATUS'].replace('Single', 0)
train_df['MARITAL_STATUS'] = train_df['MARITAL_STATUS'].replace('Other', 1)
test_df['MARITAL_STATUS'] = test_df['MARITAL_STATUS'].replace('Single', 0)
test_df['MARITAL_STATUS'] = test_df['MARITAL_STATUS'].replace('Other', 1)

# replacing age
train_df['AGE'] = train_df['AGE'].replace('Less than 30', 0.25)
train_df['AGE'] = train_df['AGE'].replace('31-45', 0.5)
train_df['AGE'] = train_df['AGE'].replace('46-65', 0.75)
train_df['AGE'] = train_df['AGE'].replace('More than 65', 1)
test_df['AGE'] = test_df['AGE'].replace('Less than 30', 0.25)
test_df['AGE'] = test_df['AGE'].replace('31-45', 0.5)
test_df['AGE'] = test_df['AGE'].replace('46-65', 0.75)
test_df['AGE'] = test_df['AGE'].replace('More than 65', 1)

# replacing due amount
train_df['DUE_AMT_JULY'] = train_df['DUE_AMT_JULY']/100000
train_df['DUE_AMT_AUG'] = train_df['DUE_AMT_AUG']/100000
train_df['DUE_AMT_SEP'] = train_df['DUE_AMT_SEP']/100000
train_df['DUE_AMT_OCT'] = train_df['DUE_AMT_OCT']/100000
train_df['DUE_AMT_NOV'] = train_df['DUE_AMT_NOV']/100000
train_df['DUE_AMT_DEC'] = train_df['DUE_AMT_DEC']/100000
test_df['DUE_AMT_JULY'] = test_df['DUE_AMT_JULY']/100000
test_df['DUE_AMT_AUG'] = test_df['DUE_AMT_AUG']/100000
test_df['DUE_AMT_SEP'] = test_df['DUE_AMT_SEP']/100000
test_df['DUE_AMT_OCT'] = test_df['DUE_AMT_OCT']/100000
test_df['DUE_AMT_NOV'] = test_df['DUE_AMT_NOV']/100000
test_df['DUE_AMT_DEC'] = test_df['DUE_AMT_DEC']/100000

# replacing paid amount
train_df['PAID_AMT_JULY'] = train_df['PAID_AMT_JULY']/12500
train_df['PAID_AMT_AUG'] = train_df['PAID_AMT_AUG']/12500
train_df['PAID_AMT_SEP'] = train_df['PAID_AMT_SEP']/12500
train_df['PAID_AMT_OCT'] = train_df['PAID_AMT_OCT']/12500
train_df['PAID_AMT_NOV'] = train_df['PAID_AMT_NOV']/12500
train_df['PAID_AMT_DEC'] = train_df['PAID_AMT_DEC']/12500
test_df['PAID_AMT_JULY'] = test_df['PAID_AMT_JULY']/12500
test_df['PAID_AMT_AUG'] = test_df['PAID_AMT_AUG']/12500
test_df['PAID_AMT_SEP'] = test_df['PAID_AMT_SEP']/12500
test_df['PAID_AMT_OCT'] = test_df['PAID_AMT_OCT']/12500
test_df['PAID_AMT_NOV'] = test_df['PAID_AMT_NOV']/12500
test_df['PAID_AMT_DEC'] = test_df['PAID_AMT_DEC']/12500

output = 'NEXT_MONTH_DEFAULT'

cols = [ f for f in train_df.columns if train_df.dtypes[ f ] != "object"]
cols = [ f for f in train_df.columns]
cols.remove( "Client_ID")
cols.remove( output )

# f = pd.melt( train_df, id_vars=output, value_vars=cols)
# g = sns.FacetGrid( f, hue=output, col="variable", col_wrap=5, sharex=False, sharey=False )
# g = g.map( sns.distplot, "value", kde=True).add_legend()

# The quantitative vars:
quant = ["Balance_Limit_V1", "AGE"]

# The qualitative but "Encoded" variables (ie most of them)
qual_Enc = cols
qual_Enc.remove("Balance_Limit_V1")
qual_Enc.remove("AGE")

logged = []
months = ["JULY","AUG","SEP","OCT","NOV","DEC"]
for i in range(0,6):
    qual_Enc.remove("PAY_" + months[i])
    train_df[ "log_PAY_" + months[i]]  = train_df["PAY_"  + months[i]].apply( lambda x: np.log1p(x) if (x>0) else 0 )
    test_df[ "log_PAY_" + months[i]]  = test_df["PAY_"  + months[i]].apply( lambda x: np.log1p(x) if (x>0) else 0 )
    logged.append("log_PAY_" + months[i] )

for ii in range(0,6):
    qual_Enc.remove("DUE_AMT_" + months[ii])
    train_df[ "log_DUE_AMT_" + months[ii]] = train_df["DUE_AMT_" + months[ii]].apply( lambda x: np.log1p(x) if (x>0) else 0 )
    test_df[ "log_DUE_AMT_" + months[ii]] = test_df["DUE_AMT_" + months[ii]].apply( lambda x: np.log1p(x) if (x>0) else 0 )
    logged.append("log_DUE_AMT_" + months[ii] )

#Visualize Matrix
features = quant + qual_Enc + logged + [output]
corr = train_df[features].corr()
plt.subplots(figsize=(30,10))
sns.heatmap( corr, square=True, annot=True, fmt=".1f" )

features = quant + qual_Enc + logged
X = train_df[features].values
y = train_df[ output ].values
submit_test = test_df[features].values


X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2)

scX = StandardScaler()
X_train = scX.fit_transform( X_train )
X_test = scX.transform( X_test )
submit_test = scX.transform( submit_test )

#--------------
# kernel SVM
#--------------
classifier1 = SVC(kernel="rbf")
classifier1.fit( X_train, y_train )

y_pred = classifier1.predict( X_test )
y_pred_submit = classifier1.predict( submit_test )

cm = confusion_matrix( y_test, y_pred )
print("Accuracy on Test Set for kernel-SVM = %.2f" % ((cm[0,0] + cm[1,1] )/len(X_test)))
scoresSVC = cross_val_score( classifier1, X_train, y_train, cv=10)
print("Mean kernel-SVM CrossVal Accuracy on Train Set %.2f, with std=%.2f" % (scoresSVC.mean(), scoresSVC.std() ))

#Save results to a file

fo = open('result.csv', 'w')
for x in np.nditer(y_pred_submit):
    fo.write(str(x))
    fo.write('\n')
fo.close()