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

