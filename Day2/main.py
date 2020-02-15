import numpy as np
import pandas as pd
import logistic_regression as lg

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

X = []
Y = []
for index, row in train_df.iterrows():
    x = []
    x.append(row['Balance_Limit_V1'])
    x.append(row['Gender'])
    x.append(row['EDUCATION_STATUS'])
    x.append(row['MARITAL_STATUS'])
    x.append(row['AGE'])
    x.append(row['PAY_JULY'])
    x.append(row['PAY_AUG'])
    x.append(row['PAY_SEP'])
    x.append(row['PAY_OCT'])
    x.append(row['PAY_NOV'])
    x.append(row['PAY_DEC'])
    x.append(row['DUE_AMT_JULY'])
    x.append(row['DUE_AMT_AUG'])
    x.append(row['DUE_AMT_SEP'])
    x.append(row['DUE_AMT_OCT'])
    x.append(row['DUE_AMT_NOV'])
    x.append(row['DUE_AMT_DEC'])
    x.append(row['PAID_AMT_JULY'])
    x.append(row['PAID_AMT_AUG'])
    x.append(row['PAID_AMT_SEP'])
    x.append(row['PAID_AMT_OCT'])
    x.append(row['PAID_AMT_NOV'])
    x.append(row['PAID_AMT_DEC'])
    x = list(map(float, x))
    y = row['NEXT_MONTH_DEFAULT']
    X.append(x)
    Y.append(y)
X_train = np.array(X).T
Y_train = np.array(Y)

X = []
for index, row in test_df.iterrows():
    x = []
    x.append(row['Balance_Limit_V1'])
    x.append(row['Gender'])
    x.append(row['EDUCATION_STATUS'])
    x.append(row['MARITAL_STATUS'])
    x.append(row['AGE'])
    x.append(row['PAY_JULY'])
    x.append(row['PAY_AUG'])
    x.append(row['PAY_SEP'])
    x.append(row['PAY_OCT'])
    x.append(row['PAY_NOV'])
    x.append(row['PAY_DEC'])
    x.append(row['DUE_AMT_JULY'])
    x.append(row['DUE_AMT_AUG'])
    x.append(row['DUE_AMT_SEP'])
    x.append(row['DUE_AMT_OCT'])
    x.append(row['DUE_AMT_NOV'])
    x.append(row['DUE_AMT_DEC'])
    x.append(row['PAID_AMT_JULY'])
    x.append(row['PAID_AMT_AUG'])
    x.append(row['PAID_AMT_SEP'])
    x.append(row['PAID_AMT_OCT'])
    x.append(row['PAID_AMT_NOV'])
    x.append(row['PAID_AMT_DEC'])
    x = list(map(float, x))
    X.append(x)

X_test = np.array(X).T
Y_test = np.zeros((1, X_test.shape[1]))

num_iterations = 1000
learning_rate = 0.5
print_cost = True

result = lg.model(X_train, Y_train, X_test, Y_test, num_iterations, learning_rate, print_cost)

fo = open('a.txt', 'w')
for line in result['Y_prediction_test'].T:
    fo.write(str(line))
    fo.write('\n')
fo.close()