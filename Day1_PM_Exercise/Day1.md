# Day 1 Exercises

Contributors: Rachel DuBose and Daniel King
University of Alabama Libraries Research Computing Services

Last updated July 10, 2026

### Learning Outcomes

At the end of this session, you should be able to:

- Access UAHPC resources
- Create a conda environment
- Run a script
- Transfer files between your local machine and UAHPC

## Accessing UAHPC

One way to connect to a Linux server is to use SSH (Secure Shell). SSH is a network protocol that allows you to securely connect to a remote server over a network.
You will need the hostname of the server, your username, and your password (or SSH key). For this session, the hostname is **uahpc.ua.edu**, the
username is your **myBamaID**, and password is your **myBama password**.

1. First, make sure you are on the VPN.

To download the client visit [here](https://ua-app01.ua.edu/software/public/vpn/showFiles).Double-click the downloaded file. It is a silent install, but you may notice some flickering of icons. Wait approximately one minute and you should see a Cisco AnyConnect box pop up.

If you have already downloaded the client, open the Cisco AnyConnect VPN client. Click “Connect” to establish a connection. If a full URL is required, enter “uavpn.ua.edu/campus”.

Authenticate with myBamaID + password + Duo method in the “Second Password” field. In the “Second Password” field, you can enter push to receive a push notification to your Duo-registered device, phone to receive a phone call to your Duo registered phone, or you can enter a passcode retrieved from the Duo app. Users must have a Duo account to access the VPN.

You may need to click Yes to allow a certificate. After this step, you’re connected!

2. Open your terminal on Linux or Mac; on Windows, use PowerShell. You will need to connect to the UAHPC server using this command:

```bash
ssh {myBamaID}@uahpc.ua.edu
```
You will be prompted to complete Duo two-factor authentication and enter your myBama password. 
If it is your first time logging on, you may be asked to accept a fingerprint; type `yes`.

## Navigating the HPC
We will first use `pwd` to see what directories we currently have and list the files:
```bash
pwd
ls
```
Next, make a directory on HPC to hold our data files, then navigate to the folder:
```bash
mkdir day1
cd day1
```
## Exercise 1
The objective of this exercise is to submit a Python script to see if we can generate matrices of user-specified size containing random floating-point values between 0 and 1, sort them, and then find the minimum and maximum numbers among all the matrices. 

First, we will see if we can obtain a minimum of 0 and a maximum of 1. 

Then we will see if we can get a minimum of 1 and maximum of 0. 

The Python script below uses the NumPy module so we can use the conda environment we created this morning.

First, we must create the file with the `nano` text editor:
```bash
nano randommat.py
```
Once the text editor window opens, copy and paste the script below into the text editor window. Then `CTRL+O`, `enter`, and `CTRL+X`.
```python
#!/usr/bin/env python
#
# Generate matrices containing random floating point numbers from 0 to 1, sort
# them, and find the minimum and maximum numbers among all the matrices.
#
# See if you can get a minimum of 0 and maximum of 1.
import numpy as np
import time

# Change these to your liking
rows = 2        # rows for each matrix
columns = 32    # columns for each matrix
matrices = 64   # number of matrices
precision = 4   # number of digits after the decimal

np.set_printoptions(precision=precision)

minimum = 0
maximum = 1

start = time.time()

for i in range(matrices):
    matrix = np.random.rand(rows, columns)
    sorted_matrix = np.sort(matrix)

    print(f"{i+1}: {matrix}")
    print(f"{i+1}: {sorted_matrix}\n")

    # Compare the first element in the sorted matrix against the minimum
    if sorted_matrix.flat[0] < minimum:
        minimum = sorted_matrix.flat[0]

    # Compare the last element in the sorted matrix against the maximum
    if sorted_matrix.flat[-1] > maximum:
        maximum = sorted_matrix.flat[-1]

end = time.time()

elements = rows * columns * matrices # total number of elements across all matrices
print(f"Maximum of {elements} numbers: {maximum:.{precision}f}")
print(f"Minimum of {elements} numbers: {minimum:.{precision}f}")
print(f"Finished in {end - start:.{precision}f} seconds")
```
To submit the job, we need a batch script. 

This script uses NumPy which is not built-in and can be utilized through a Conda environment. 

We will use the Conda environment we created this morning and activate the environment within the batch script. 
```bash
nano random.batch
```
Once the text editor window opens, copy and paste the script below into the text editor window. Then `CTRL+O`, `enter`, and `CTRL+X`.
```bash
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=8G
#SBATCH --job-name=random-matrices
#SBATCH --partition=main
#SBATCH --qos=main
#SBATCH -e errors.%A
#SBATCH -o output.%A

# Environment setup
module load miniconda3/base
conda activate day1

# Execution
python ./randommat.py
```
Now let's run the script:
```bash
sbatch random.batch
```

We can verify the script ran:
```bash 
ls
```
You should see an errors file and an output file. To view them, use:
```bash
cat errors.%A
cat output.%A
```
where `%A` is the job number.

How about trying a minimum of 1 and a maximum of 0? We must first adjust the Python script with `nano`:
```bash
nano randommat.py
```
With the text editor window open, change to:
```python
minimum = 1
maximum = 0
```
Type `CTRL+O`, `enter`, and `CTRL+X` to save and exit the text editor.

We can now submit the job again with:
```bash
sbatch random.batch
```
To view the output and errors files:
```bash
cat errors.%A
cat output.%A
```
where `%A` is the job number.

## Exercise 2
For Exercise 2, we will be doing something a little more complex and we will need more packages. To accomplish this, we will create a new conda environment.

Make sure we are no longer in the `day1` directory:
```bash
cd ..
pwd
```
We should see:
```bash
[myBamaID@uahpc-login001 ~]$
```
Now, load the module:
```bash 
module load miniconda3/base
```
And create the conda environment called `healthenv`:
```bash
conda create --name healthenv
conda activate healthenv
conda install -c conda-forge python numpy matplotlib seaborn pandas scipy -y
conda deactivate
```
For this exercise, we are interested in examining how health outcomes are related to economics.

The [dataset](https://ourworldindata.org/grapher/life-expectancy-vs-health-expenditure) is built-in to the Seaborn package.

This dataset is licensed under a Creative Commons Attribution 4.0 International (CC BY 4.0) license.

It provides the period life expectancy at birth, in a given year as well as health expenditures which include all financing schemes and covers all aspects of healthcare. This data is adjusted for inflation and differences in living costs between countries.

We plan to use the dataset to explore the following questions:
1. How has healthcare spending changed over time?
2. How has life expectancy changed over time?
3. Which countries have the highest life expectancy?
4. Is there a correlation between healthcare spending and life expectancy?

We will use the Python script below. We will create it on the UAHPC with `nano`.
```bash
nano healthexp.py
```
Once the text editor window opens, copy and paste the script below into the text editor window. Then `CTRL+O`, `enter`, and `CTRL+X`.

```python
#import libraries
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

#Import life expectancy data
health_exp = sns.load_dataset('healthexp')

#lineplot of spending_USD by year
sns.lineplot(x = 'Year', y = 'Spending_USD', data = health_exp, hue = 'Country', errorbar =None, markers = True)
sns.set(rc={'figure.figsize': (15,10)}) #set length of x and y axis
sns.set_style("whitegrid") #white background
sns.set_context("talk") #increase font size
plt.title("Spending on Healthcare by Year")
plt.xlabel('Year')
plt.ylabel('Healthcare Spending in USD') 
plt.savefig('spending.png')
plt.close()

#Create lineplot of life expectancy by year
sns.lineplot(x = 'Year', y = 'Life_Expectancy', data = health_exp, hue = 'Country', errorbar =None, markers = True)
sns.set(rc={'figure.figsize': (15,10)}) #set length of x and y axis
sns.set_style("whitegrid") #white background
sns.set_context("talk") #increase font size
plt.title("Life Expectancy over time")
plt.xlabel('Year')
plt.ylabel('Life Expectancy') 
plt.savefig('life_expect.png')
plt.close()

#Sort by life expectancy and display top 5 countries
top_5_life_exp = health_exp.groupby('Country')['Life_Expectancy'].max().sort_values(ascending=False).head(5)
print("\nTop 5 Countries by Maximum Life Expectancy:")
print(top_5_life_exp)

#Create scatter plot showing correlation between healthcare spending and life expectancy
sns.set(rc={'figure.figsize': (12,8)})
sns.scatterplot(x='Spending_USD', y='Life_Expectancy', data=health_exp, hue='Country', s=100, alpha=0.6)
sns.regplot(x='Spending_USD', y='Life_Expectancy', data=health_exp, scatter=False, color='red', label='Trend Line')
plt.title("Correlation between Healthcare Spending and Life Expectancy")
plt.xlabel('Healthcare Spending (USD)')
plt.ylabel('Life Expectancy (years)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('correlation.png')
plt.close()

#Calculate and display correlation statistics
correlation = health_exp['Spending_USD'].corr(health_exp['Life_Expectancy'])
print(f"\nPearson Correlation Coefficient: {correlation:.4f}")

#Additional statistics using linear regression
slope, intercept, r_value, p_value, std_err = stats.linregress(health_exp['Spending_USD'].dropna(), 
                                                                health_exp['Life_Expectancy'].dropna())
print(f"R-squared: {r_value**2:.4f}")
print(f"P-value: {p_value:.4e}")
print(f"Slope: {slope:.6f}")
print(f"Standard Error: {std_err:.6f}")
```
We will also need a batch script. The script below loads the conda module and activates the conda environment we previously created. 

Again, use `nano`:
```bash
nano health.batch
```
Once the text editor window opens, copy and paste the script below into the text editor window. Then `CTRL+O`, `enter`, and `CTRL+X`.
```bash
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=8G
#SBATCH --job-name=healthanalysis
#SBATCH --partition=main
#SBATCH --qos=main
#SBATCH -e health_errors.%A
#SBATCH -o health_output.%A

# Environment setup
module load miniconda3/base
conda activate healthenv

# Execution
python ./healthexp.py
```
Now, submit the script with:
```bash
sbatch health.batch
```
To view the output and errors files:
```bash
cat health_errors.%A
cat health_output.%A
```
where `%A` is the job number.

Verify that the plots were produced. 
```bash
ls
```
We can use SFTP to move the files from UAHPC to our local computer.
In new terminal, type the following and complete authentication steps:
```bash
sftp myBamaID@uahpc.ua.edu
```
Then use the `get` command to retrieve the .png files.
```bash
get spending.png C:\Users\myBamaID\Downloads
get life_expect.png C:\Users\myBamaID\Downloads
get correlation.png C:\Users\myBamaID\Downloads
```
Open them on your own computer to view.
Don't close the SFTP terminal window as we will use it in the next exercise as well.

## Exercise 3
For Exercise 3, we will be doing something different and using R instead of Python. 
We will need to load a new module.
```bash
module avail math/R
```
Now we load the selected version of R.
```bash
module load math/R/4.5.2
```
For this exercise, we are using the mtcars dataset which was from the 1974 Motor Trend US magazine, and comprises fuel consumption and 10 aspects of automobile design and performance for 32 automobiles (1973-74 models). This dataset is in the public domain under CC0.

We plan to use the dataset to explore the following questions:
1. Which cars have the best fuel economy?
2. Is fuel economy significantly different between transmission types?
3. What is the correlation between vehicle weight and fuel economy?

We will use the R script below. We will create it on the UAHPC with `nano`.
```bash
nano mtcars_analysis.R
``` 
Copy and paste the following:

```r
# Load mtcars dataset (built-in to R)
data(mtcars)

# Filter for high mpg (>20) and high horsepower (>100)
mtcars_high_mpg <- subset(mtcars, mpg > 20 & hp > 100)

cat("\n Cars with mpg > 20 and hp > 100 \n")
print(mtcars_high_mpg)

# Perform t-test between automatic and manual transmission (am = 0 for automatic, am = 1 for manual)
cat("T-Test: mpg by Transmission Type \n")
cat("H0: No difference in mpg between automatic and manual transmissions\n\n")

t_test_result <- t.test(mpg ~ am, data = mtcars, alternative = "two.sided")
print(t_test_result)

if (t_test_result$p.value < 0.05) {
  cat("\nConclusion: Reject null hypothesis (p < 0.05)\n")
  cat("There IS a significant difference in mpg between transmission types.\n\n")
} else {
  cat("\nConclusion: Fail to reject null hypothesis (p >= 0.05)\n")
  cat("There is NO significant difference in mpgg between transmission types.\n\n")
}


# Linear regression (mpg vs weight)
cat("Linear Regression: mpg vs. weight\n\n")

fit <- lm(mpg ~ wt, data = mtcars)

cat("Model Summary:\n")
print(summary(fit))

cat("\n\nCoefficients:\n")
print(coef(fit))

cat("\n\nInterpretation:\n")
cat(sprintf("Intercept: %.2f mpg (expected mpg when weight = 0)\n", coef(fit)[1]))
cat(sprintf("Slope: %.2f mpg per 1000 lbs (mpg decreases by %.2f for each 1000 lb increase)\n", 
            coef(fit)[2], abs(coef(fit)[2])))

# Calculate correlation coefficient
correlation <- cor(mtcars$wt, mtcars$mpg)
cat(sprintf("\nPearson Correlation: %.4f\n", correlation))

# Create scatter plot with regression line
png("mpg_vs_weight.png", width = 800, height = 600, res = 100, type='cairo')
plot(mtcars$wt, mtcars$mpg, 
     main = "mpg vs. weight", 
     xlab = "Weight (1000 lbs)", 
     ylab = "Miles Per Gallon",
     pch = 19,
     col = "blue")
abline(fit, col = "red", lwd = 2)
legend("topright", legend = c("Observed Data", "Regression Line"), 
       col = c("blue", "red"), pch = c(19, NA), lty = c(NA, 1), lwd = c(NA, 2))
dev.off()
```
We will also need a batch script. The script below loads the R module. 

Again, use `nano`. Once the text editor window opens, copy and paste the script below into the text editor window. Then `CTRL+O`, `enter`, and `CTRL+X`.
```bash
nano mtcars.batch
```
```bash
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=4G
#SBATCH --job-name=mtcars-analysis
#SBATCH --partition=main
#SBATCH --qos=main
#SBATCH -e mtcars_errors.%A
#SBATCH -o mtcars_output.%A

# Environment setup
module load math/R/4.5.2

# Execution
Rscript mtcars_analysis.R
```
We can then submit the batch script using:
```bash
sbatch mtcars.batch
```
To view the output and errors files:
```bash
cat mtcars_errors.%A
cat mtcars_output.%A
```
where `%A` is the job number.

Verify that the plot was produced.
```bash
ls
```
In the SFTP terminal window, use the `get` command to retrieve the .png file.
```bash
get mpg_vs_weight.png C:\Users\myBamaID\Downloads
```
