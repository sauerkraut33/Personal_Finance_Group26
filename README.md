# Personal_Finance_Group26

Goal: 1. Visualizing how financial pressures evolve from early adulthood through mid-life.
      2. Create a predictive model such that we can predict financial pressures of mid-life people from their early adulthood financial situation.

Definition: Financial Pressures: a combination value of Mortgage Debt, Student Loan Debt Credit Card Debt and Line of Credit Debt (shorten as PWDs)

Process:
1. Data Processing:
    1) Find related variables by checking correlations between variables and PWDs
    2) Clean variables (process abnormal data, record them and show how to deal with them)
    3) Produce coefficients needed by modelling section
2. Modelling:
    Create Predictive Model:
        1) Define financial pressure as a combination of PWDs such as a pressure index I = aPWD1+bPWD2+cPWD3...
        2) Create the model as a formula of financial pressure by related variables
        3) Derive coefficients needed from data processing section
3. Visualization:
    1) Correlation graph between each related variable and each related PWD
    2) ... and financial pressure
    3) variation trend graph of each PWD according to age
    4) ... of financial pressure ...



## Data Processing Usage Guide
The data processing workflow consists of three sequential scripts. Please follow the steps in order.

**1)Data Loading**
Before running `data_load.py`, update the file path string inside the script so that it points to your local copy of `personal_finance_dataset.xlsx`.
This file must exist on your machine. The script reads the Excel dataset and prepares it for further processing.
Run `data_load.py` after updating the path.

**2)Outlier Removal**
After successfully running the data loading step, execute `data_remove_outlier.py`.
This script uses Inerquartile Range method to remove statistical outliers in net worth values from the processed dataset and generates a cleaned file:
`personal_finance_cleaned.csv`
This cleaned dataset will be used in the next step.

**3)Rank Correlation Analysis**
Finally, run `rank_correlation.py`.
This script computes rank correlations between selected variables and produces a visualization plot as output.


## Modelling Usage Guide
The modelling workflow is located in the modelling folder. Please follow the steps in order.

**1)Remodelling**
First, run `remodelling_data.py`.
This script reads the cleaned dataset generated in the data processing stage and prepares the final input file for modelling.
After execution, it will generate:
`personal_finance_model_input.csv`
This file will be used in the next step.

**2)Rank Correlation Analysis**
Next, run `rank_correlation.py` inside the modelling folder.
This script computes rank correlations based on the model input dataset and produces a visualization plot as output.
