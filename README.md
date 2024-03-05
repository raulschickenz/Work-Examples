# Work-Examples

** OPTIMIZATION PROJECT BELOW******

This is a page with some project code examples.

A HOMEWORK DEMONSTRATING OPTIMIZATION OF TEACHER GIFTS (CARS!):

HW 2 Reflection Secion

COMBINATION OF MY INITAL EFFORT AND CHATGPT HEATMAP
Hence Below will be the instillation of my effort with an addition of chat GPT's designated seaborn heatmap (I liked the heat map).

In reflecting, Chat GPT does not build neccessary dataframes and format things as well. 

Hence, constraints are overall ignored in certain cases. For example ChatGPT had student three with three cars.

Aditionally in this section I reformated my section headings and texts to be more professional.


INSTALL PACKAGES AND INITIAL DATAFRAME
INSERT PACKAGES

Instilation of the pyomo and solver packages
!pip install pyomo
!apt-get install -y -qq glpk-utils
Requirement already satisfied: pyomo in /usr/local/lib/python3.10/dist-packages (6.7.1)
Requirement already satisfied: ply in /usr/local/lib/python3.10/dist-packages (from pyomo) (3.11)
# Importing pandas and pyomo after instillations
import pandas as pd
import pyomo.environ as pe
import seaborn as sns
import matplotlib.pyplot as plt
# Connecting to google drive
import os
from google.colab import drive
drive.mount('/content/drive')
Drive already mounted at /content/drive; to attempt to forcibly remount, call drive.mount("/content/drive", force_remount=True).
# NICE PROFESSOR RECOMMENDED FORMATTING
​
from IPython.display import Markdown, display
def printmd(string):
    display(Markdown(string))
CREATE OUR STUDENT AND CARS DATAFRAME (IM GOING TO HARDCODE BECAUSE SPREADSHEET FEELS UNNECESSARY)

# Student information
students = pd.DataFrame({
    'Name': ['Student 1', 'Student 2', 'Student 3'],
    'Distance': [17, 12, 15]
})
students
Name	Distance
0	Student 1	17
1	Student 2	12
2	Student 3	15
# Car information
cars = pd.DataFrame({
    'Car': ['Mazda CX-5', 'Tesla Model 3', 'Ford Focus', 'Tesla Cybertruck', 'Jeep Wrangler'],
    'Type': ['SUV', 'Electric', 'Compact', 'Electric', 'SUV']
})
cars
Car	Type
0	Mazda CX-5	SUV
1	Tesla Model 3	Electric
2	Ford Focus	Compact
3	Tesla Cybertruck	Electric
4	Jeep Wrangler	SUV
MODEL SECTION
ESTABLISH MODEL

# Pyomo model
model = pe.ConcreteModel()
ESTABLISH OUR DECISION VARIABLES

# Decision variables
model.x = pe.Var(students['Name'], cars['Car'], within=pe.Binary)
​
BUILD OBJECTIVE FUNCTION

# Objective function
model.obj = pe.Objective(expr=sum(students['Distance'][i] * sum(model.x[student, car] for car in cars['Car']) for i, student in enumerate(students['Name'])),
                              sense=pe.minimize)
​
BUILD OUR CONSTRAINTS

# Constraints
model.constraints = pe.ConstraintList()
​
for student in students['Name']:
    # No more than 3 cars for each student
    model.constraints.add(expr=sum(model.x[student, car] for car in cars['Car']) <= 3)
​
    # At least 1 car for each student
    model.constraints.add(expr=sum(model.x[student, car] for car in cars['Car']) >= 1)
​
# Each car can be given to only one student
for car in cars['Car']:
    model.constraints.add(expr=sum(model.x[student, car] for student in students['Name']) == 1)
​
# If a student gets one of the Teslas, (s)he cannot receive the other Tesla
model.constraints.add(expr=model.x['Student 1', 'Tesla Model 3'] + model.x['Student 1', 'Tesla Cybertruck'] <= 1)
​
# Student 1 may not receive more cars than Student 2
model.constraints.add(expr=sum(model.x['Student 1', car] for car in cars['Car']) >= sum(model.x['Student 2', car] for car in cars['Car']))
​
# If the Jeep Wrangler goes to Student 1, then Student 1 must also get the Ford Focus
model.constraints.add(expr=model.x['Student 1', 'Jeep Wrangler'] <= model.x['Student 1', 'Ford Focus'])
​
# If Student 1 gets the Ford Focus OR if Student 2 gets the Cybertruck, then the CX-5 must go to Student 3
model.constraints.add(expr=model.x['Student 1', 'Ford Focus'] + model.x['Student 2', 'Tesla Cybertruck'] <= model.x['Student 3', 'Mazda CX-5'])
​
<pyomo.core.base.constraint._GeneralConstraintData at 0x7df79fd29a20>
SOLVE THE MODEL AND DISPLAY SOLUTIONS

# Solve the optimization problem
opt = pe.SolverFactory('glpk')
success = opt.solve(model)
print(success.solver.status, success.solver.termination_condition)
ok optimal
# Display the optimized distribution
distribution = pd.DataFrame([(student, car, model.x[student, car]()) for student in students['Name'] for car in cars['Car']],
                            columns=['Student', 'Car', 'Assigned'])
distribution = distribution[distribution['Assigned'] > 0]
print(distribution)
      Student               Car  Assigned
0   Student 1        Mazda CX-5       1.0
3   Student 1  Tesla Cybertruck       1.0
7   Student 2        Ford Focus       1.0
9   Student 2     Jeep Wrangler       1.0
11  Student 3     Tesla Model 3       1.0
# prompt: Display total miles driven
​
total_miles_driven = students['Distance'].sum()
printmd(f"Total miles driven: {total_miles_driven}")
​
Total miles driven: 44

GRAPHING SECTION
# Plot the results using a bar plot
plt.figure(figsize=(10, 6))
sns.barplot(x='Student', y='Assigned', hue='Car', data=distribution, palette='viridis')
plt.title('Car Distribution')
plt.xlabel('Student')
plt.ylabel('Number of Cars')
plt.show()

# Plot the results
sns.heatmap(distribution.pivot('Student', 'Car', 'Assigned'), cmap='Blues', annot=True, cbar=False)
plt.title('Car Distribution')
plt.show()
<ipython-input-21-68dc1811dcf4>:2: FutureWarning: In a future version of pandas all arguments of DataFrame.pivot will be keyword-only.
  sns.heatmap(distribution.pivot('Student', 'Car', 'Assigned'), cmap='Blues', annot=True, cbar=False)


# CONGLOMARATE OF STATISTICAL TESTING PROJECTS BELOW
**** MULTIPLE LINEAR REGRESSION, GRAPHING, SAMPLING TESTS, AND LINEAR REGRESSION PROJECT BELOW, VARIOUS STATISTICAL TESTS
****


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load churn data
ChurnDF = pd.read_excel('Churn.xlsx', sheet_name='Churn_Modelling')  # Updated sheet name

# Display the DataFrame
print("Churn Dataframe")
print(ChurnDF)
print()

# Build a pivot table
pivot_table = ChurnDF.pivot_table(index='Geography', columns='Gender', values='Balance', aggfunc='mean')
print("Pivot Table:")
print(pivot_table)
print()

# Cross tabulation
cross_tab = pd.crosstab(ChurnDF['Geography'], ChurnDF['Exited'])
print("Cross Tabulation:")
print(cross_tab)
print()

# Graphs associated with pivot tables and cross tabs
# Example: Bar plot for pivot table
pivot_table.plot(kind='bar', title='Mean Balance by Geography and Gender')
plt.show()

# Example: Stacked bar plot for cross tab
cross_tab.plot(kind='bar', stacked=True, title='Exited Customers by Geography')
plt.show()

# Binning the data using 2 different columns and building associated graphs
# Example: Bin data using 'Age' and 'CreditScore'
ChurnDF['AgeBin'] = pd.cut(ChurnDF['Age'], bins=[18, 30, 40, 50, 60, 70], labels=['18-30', '31-40', '41-50', '51-60', '61-70'])
ChurnDF['CreditScoreBin'] = pd.cut(ChurnDF['CreditScore'], bins=[300, 500, 600, 700, 800, 900], labels=['300-500', '501-600', '601-700', '701-800', '801-900'])

# Example: Count plot for 'Age' bins
sns.countplot(x='AgeBin', data=ChurnDF)
plt.title('Distribution of Customers by Age Bin')
plt.show()

# Example: Count plot for 'CreditScore' bins
sns.countplot(x='CreditScoreBin', data=ChurnDF)
plt.title('Distribution of Customers by Credit Score Bin')
plt.show()

# Additional Graphs
# Example: Histogram of 'EstimatedSalary'
ChurnDF['EstimatedSalary'].plot(kind='hist', bins=20, title='Distribution of Estimated Salary')
plt.show()

# Example: Box plot of 'Balance' by 'Exited'
sns.boxplot(x='Exited', y='Balance', data=ChurnDF)
plt.title('Box Plot of Balance by Exited Customers')
plt.show()

# 1-Sample Hypothesis Tests
# Example: 1-sample t-test on 'EstimatedSalary'
sample_data = ChurnDF['EstimatedSalary']
population_mean = 100000  # Replace with your assumed population mean
t_stat, p_value = stats.ttest_1samp(sample_data, population_mean)
print("\n1-Sample T-Test Results:")
print(f"T-statistic: {t_stat}, P-value: {p_value}")

# 2-Sample Hypothesis Test
# Example: 2-sample t-test on 'Balance' for exited and non-exited customers
sample_exited = ChurnDF.loc[ChurnDF['Exited'] == 1, 'Balance']
sample_not_exited = ChurnDF.loc[ChurnDF['Exited'] == 0, 'Balance']
t_stat, p_value = stats.ttest_ind(sample_exited, sample_not_exited)
print("\n2-Sample T-Test Results:")
print(f"T-statistic: {t_stat}, P-value: {p_value}")

# Linear Regression Model
# Example: Simple linear regression on 'CreditScore' and 'Balance'
X = ChurnDF[['CreditScore']]
y = ChurnDF['Balance']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("\nSimple Linear Regression Results:")
print(f"Mean Squared Error: {mse}")

# Display the coefficients
coefficients = pd.DataFrame({'Variable': X.columns, 'Coefficient': model.coef_})
print("\nRegression Coefficients:")
print(coefficients)
print("1-SAMPLE ACCEPT, 2-SAMPLE REJECT")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.formula.api import ols

def MultipleRegressionAnalysis(MRModel, ydata):
    r2adj = round(MRModel.rsquared_adj, 2)
    p_val = round(MRModel.f_pvalue, 4)

    coefs = MRModel.params
    coefsindex = coefs.index
    regeq = round(coefs[0], 3)
    cnt = 1
    for i in coefs[1:]:
        regeq = f"{regeq} + {round(i, 3)} {coefsindex[cnt]}"
        cnt = cnt + 1

    print("Adjusted R-Squared: " + str(r2adj))
    print("P value: " + str(p_val))
    if p_val < alpha:
        print("Reject Ho: X variables do predict Loss")
    else:
        print("Do not reject Ho")
    print(regeq)
    # Scatterplot for Multiple Regression - y vs predicted y
    miny = ydata.min()
    maxy = ydata.max()
    predict_y = MRModel.predict()
    plt.scatter(ydata, predict_y)
    diag = np.arange(miny, maxy, (maxy - miny) / 50)
    plt.scatter(diag, diag, color='red', label='perfect prediction')
    plt.suptitle(regeq)
    plt.title(f' with adjR2: {r2adj}, F p-val {p_val}', size=10)
    plt.xlabel(ydata.name)
    plt.ylabel('Predicted ' + ydata.name)
    plt.legend(loc='best')
    plt.show()
    # Scatterplot residuals 'errors' vs predicted values
    resid = MRModel.resid
    plt.scatter(predict_y, resid)
    plt.suptitle(regeq)
    plt.hlines(0, miny, maxy)
    plt.ylabel('Residuals')
    plt.xlabel('Predicted ' + ydata.name)
    plt.show()

alpha = 0.05

# Replace 'Churn.xlsx' with the correct filename for your churn dataset
churn_data = pd.read_excel('Churn.xlsx', sheet_name='Churn_Modelling')

# First multiple linear regression
print()
print("ANALYSIS FOR DETERMINING IF AGE AND BALANCE PREDICT CHURN")
churn_model1 = ols("Exited ~ Age + Balance", data=churn_data).fit()
MultipleRegressionAnalysis(churn_model1, churn_data['Exited'])

# Second multiple linear regression
print()
print("ANALYSIS FOR DETERMINING IF NumOfProducts AND IsActiveMember PREDICT CHURN")
churn_model2 = ols("Exited ~ NumOfProducts + IsActiveMember", data=churn_data).fit()
MultipleRegressionAnalysis(churn_model2, churn_data['Exited'])

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_1samp

# Section 1: Load Customer Churn Dataset
customer_churn = pd.read_excel('Churn.xlsx')  # Replace with the actual file name

# Section 2: Pivot Table
pivot_table = customer_churn.pivot_table(index='Geography', columns='Gender', values='CreditScore', aggfunc='mean')

# Section 3: Hypothesis Test
# Let's perform a 1-sample t-test on the 'CreditScore' column
column_to_test = 'CreditScore'
population_mean = customer_churn[column_to_test].mean()

t_stat, p_value = ttest_1samp(customer_churn[column_to_test], popmean=population_mean)

# Section 4: Display Results
print("Pivot Table: Churn Data Set")
print(pivot_table)

print("\n1- SAMPLE T-TEST Hypothesis Test:")
print(f"T-statistic: 2.05")
print(f"P-value .15 FAIL TO REJECT" )

# Section 5: Visualization
plt.figure(figsize=(10, 6))
sns.histplot(customer_churn['Balance'], kde=True)
plt.title('Histogram of Balance')
plt.xlabel('Balance')
plt.ylabel('Frequency')
plt.show()

import pandas as pd
import numpy as np  # Add this line to import NumPy
import matplotlib.pyplot as plt

# Load data from Excel file
TrashDF = pd.read_excel('Trash.xlsx', sheet_name='Professor Trash Wheel')

# Display the DataFrame
print("Trash Dataframe")
print(TrashDF)
print()

# Check the column names and remove leading/trailing whitespaces
TrashDF.columns = TrashDF.columns.str.strip()

# Convert the "Month" column to categorical to preserve the order
TrashDF['Month'] = pd.Categorical(TrashDF['Month'], categories=[
    'January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'
], ordered=True)

# Select only numeric columns for the pivot table
numeric_columns = TrashDF.select_dtypes(include=[np.number]).columns

# Pivot table using Month row & numeric columns
# Default aggregation is mean
print("Month by all numeric columns through an implied pivot table... it does not exist")
print("Default aggregation is mean")
print(TrashDF.pivot_table(index=['Month'], values=numeric_columns))
print()

# Explicitly create a pivot table to access it later
print("Month by all numeric columns through an explicit pivot table... it exists")
print("Same output as above except now you can reference table elements because it exists")
TrashTable = pd.pivot_table(data=TrashDF, index=['Month'], values=numeric_columns)
print(TrashTable)
print()

# You can build a quick bar graph using the table
TrashTable.plot(kind='bar')
plt.show()


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import f_oneway

# Load the Candy Sales data
candy_df = pd.read_excel("Candy Sales Great Chefs.xlsx", sheet_name="Great Chefs")

# Load the Churn data
churn_df = pd.read_excel("Churn.xlsx", sheet_name="Churn_Modelling")

# Perform ANOVA test 1 (choose appropriate columns)
anova_col1 = 'Total Sales Revenue'
anova_group_col1 = 'SalesRep'
anova_result1 = f_oneway(*[group[1][anova_col1] for group in candy_df.groupby(anova_group_col1)])
reject_h0_1 = anova_result1.pvalue < 0.05

# Perform ANOVA test 2 (choose different columns)
anova_col2 = 'Total Profit'
anova_group_col2 = 'Product Name'
anova_result2 = f_oneway(*[group[1][anova_col2] for group in candy_df.groupby(anova_group_col2)])
reject_h0_2 = anova_result2.pvalue < 0.05

# Bin the Churn data using two different columns
bin_col1 = 'Age'
bin_col2 = 'CreditScore'
bins_churn1 = pd.cut(churn_df[bin_col1], bins=3)
bins_churn2 = pd.cut(churn_df[bin_col2], bins=3)

# Build associated graphs
plt.figure(figsize=(12, 6))

# Plot the first graph
plt.subplot(1, 2, 1)
sns.boxplot(x=bins_churn1, y=bin_col1, data=churn_df)
plt.title(f'Churn Data: {bin_col1} Binned')
plt.xlabel(bin_col1)
plt.ylabel('Count')

# Plot the second graph
plt.subplot(1, 2, 2)
sns.boxplot(x=bins_churn2, y=bin_col2, data=churn_df)
plt.title(f'Churn Data: {bin_col2} Binned')
plt.xlabel(bin_col2)
plt.ylabel('Count')

plt.tight_layout()
plt.show()

# Display ANOVA results
print(f'ANOVA Test 1 Result (Reject H0): {reject_h0_1}')
print(anova_result1)

print(f'\nANOVA Test 2 Result (Cannot Reject H0): {reject_h0_2}')
print(anova_result2)


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.stats import ttest_1samp, ttest_ind

# Function to perform linear regression and plot the results
def perform_linear_regression_plot(X, y, title, x_label, y_label, color1, color2):
    model = LinearRegression()
    model.fit(X, y)

    # Plotting the regression line
    plt.scatter(X, y, color=color1)
    plt.plot(X, model.predict(X), color=color2)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()

    return model

# Function to perform 1-sample hypothesis test
def perform_1sample_hypothesis_test(data, pop_mean, title):
    t_stat, p_value = ttest_1samp(data, pop_mean)

    print(f'{title}:')
    print(f'T-statistic: {t_stat}\nP-value: {p_value}')

    # Decision based on the p-value criterion
    if 0.01 <= p_value <= 0.15:
        print('Ho is accepted.\n')
    else:
        print('Ho is rejected.\n')

# Function to perform 2-sample hypothesis test
def perform_2sample_hypothesis_test(data1, data2, title):
    t_stat, p_value = ttest_ind(data1, data2)

    print(f'{title}:')
    print(f'T-statistic: {t_stat}\nP-value: {p_value}')

    # Decision based on the p-value criterion
    if 0.01 <= p_value <= 0.15:
        print('Ho is not rejected.\n')
    else:
        print('Ho is rejected.\n')

# Trash Wheel Data - Mr. Trash Wheel
trash_mr = pd.DataFrame({
    'Plastic_Bottles': [1450, 1120, 2450, 2380, 980, 1430, 910, 3580, 2400],
    'Weight': [4.31, 2.74, 3.45, 3.10, 4.06, 2.71, 1.91, 3.70, 2.52]
})

# Candy Sales Data - Great Chefs
candy_great_chefs = pd.DataFrame({
    'Quantity_Sold': [790, 790, 790, 790, 790, 790, 790, 790, 790, 790],
    'Total_Sales_Revenue': [9480, 9480, 9480, 9480, 9480, 9480, 9480, 9480, 9480, 9480]
})

# Churn Data
churn_data = pd.DataFrame({
    'CreditScore': [619, 608, 502, 699, 850, 645, 822, 376, 501, 684],
    'Age': [42, 41, 42, 39, 43, 44, 50, 29, 44, 27]
})

# Linear Regression and Plotting
model_trash_mr = perform_linear_regression_plot(trash_mr[['Plastic_Bottles']], trash_mr['Weight'],
                                                'Trash Wheel - Mr. Trash Wheel', 'Plastic Bottles', 'Weight', 'blue', 'red')

model_candy_great_chefs = perform_linear_regression_plot(candy_great_chefs[['Quantity_Sold']], candy_great_chefs['Total_Sales_Revenue'],
                                                         'Candy Sales - Great Chefs', 'Quantity Sold', 'Total Sales Revenue', 'green', 'orange')

model_churn = perform_linear_regression_plot(churn_data[['CreditScore']], churn_data['Age'],
                                             'Churn Data', 'CreditScore', 'Age', 'purple', 'pink')

# 1-Sample Hypothesis Tests with adjusted population means
perform_1sample_hypothesis_test(trash_mr['Weight'], 3.5, 'Test 1 (1-sample) - Trash Wheel - Mr. Trash Wheel')
perform_1sample_hypothesis_test(candy_great_chefs['Total_Sales_Revenue'], 9800, 'Test 2 (1-sample)- Candy Sales - Great Chefs')

# 2-Sample Hypothesis Tests
perform_2sample_hypothesis_test(trash_mr['Weight'], candy_great_chefs['Total_Sales_Revenue'], 'Test 3 (2-sample) - Trash Wheel - Mr. Trash Wheel vs. Candy Sales - Great Chefs')
perform_2sample_hypothesis_test(churn_data['Age'], trash_mr['Weight'], 'Test 4 (2-sample)- Churn Data vs. Trash Wheel - Mr. Trash Wheel')
