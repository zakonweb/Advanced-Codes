"""
Problem to be solved using python:
In this problem, we will simulate expected outcome of the upcoming general election in Pakistan. 

In Pakistan, in the National Assembly, we have total of 342 seats and one has to secure 172 seats to establish majority and thereby form the National Government.

Though there are several political parties in Pakistan, we will list only some of them with their past 2018 scores without naming them with their true names. Consider a dictionary {′A′ : 149,′ B′ : 82,′ C′ : 54,′ D′ : 15,′ E′ : 3,′ F ′ : 1,′ G′ : 7,′ H′ : 5,′ I′ : 5,′ J′ : 4,′ K′ : 1,′ L′ : 1,′ M′ : 13,′ N′ : 2}, which has hypothetical names of all political parties and actual NA seats won in the 2018 elections. 

We construct a DataFrame which consists of two columns such that the first column has names of the political parties and the second column has corresponding NA seats as given above. Label the columns with appropriate titles. 

For simulation purpose, we define a function that generates random number representing number of seats won by each political party in such a way that the total sum of all the seats won by all political parties equals 342.

The above 2018 data suggests that for any party, it is less likely to score seats above 100 and comparatively more chances to score between 50-100 and even more to score less than 50 seats. Based on these observations, we assign probabilities to each randomly generated number of seats. 

We define a function that assigns probability to each score with following details: if score is greater than 100 its probability is 0.08, if the score is between 50-100 the probability is 0.15 and if the score is less than 50 the probability is 0.77. 

We append two more columns of randomly generated scores and their corresponding probabilities respectively to the previously defined DataFrame. Repeat the process of allocating random seats and assigning probabilities to each party 100 times. Subsequently, calculate the averages of the seats and probabilities for each party over 100 runs. 
Represent 2018 and found averaged results over 100 runs with some appropriate graphs. The following itemized list shows the required items in your code.
1. There should be a DataFrame object by the name of df where you store and work with all of your data.
2. There must be a function by the name of generate seats() which allocates random seats to each party such that the sum is 342.
3. There must be a function by the name of assign prob() which assigns probabilities to randomly generated seats as per description given above.
4. Generate a figure and divide it into four axes such that first axis shows graph of data from 2018, second graph shows that average data over 100 runs, third graphs shows the average probabilities and fourth graph combines previous three graphs. For each graph, on x axis always plot name of the political party. You may decide any type of the graph such as line, bar, etc. Each graph should have proper labels, markers, markersize, legend, etc.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Initial data
data = {'Party': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N'],
        '2018_Seats': [149, 82, 54, 15, 3, 1, 7, 5, 5, 4, 1, 1, 13, 2]}

df = pd.DataFrame(data)

def generate_seats():
    seats = np.random.multinomial(342, [1/14]*14)
    return seats

def assign_prob(seats):
    probabilities = [0.08 if s > 100 else 0.15 if 50 <= s <= 100 else 0.77 for s in seats]
    return probabilities

# Generate seats and probabilities
# This code creates a DataFrame df with the given data, simulates 100 runs of the election using the specified probability distribution, calculates the averages for each party, and plots the required graphs.
# Simulate 100 runs
for i in range(100):
    seats = generate_seats()
    probs = assign_prob(seats)
    df[f"Run_{i+1}_Seats"] = seats
    df[f"Run_{i+1}_Probs"] = probs

# Calculate averages
df['Average_Seats'] = df.iloc[:, 2:102].mean(axis=1)
df['Average_Probs'] = df.iloc[:, 102:202].mean(axis=1)

# Plotting
fig, axs = plt.subplots(2, 2, figsize=(12, 8))

# 2018 data
axs[0, 0].bar(df['Party'], df['2018_Seats'])
axs[0, 0].set_title('2018 Seats')
axs[0, 0].set_xlabel('Party')
axs[0, 0].set_ylabel('Seats')

# Average data over 100 runs
axs[0, 1].bar(df['Party'], df['Average_Seats'])
axs[0, 1].set_title('Average Seats over 100 Runs')
axs[0, 1].set_xlabel('Party')
axs[0, 1].set_ylabel('Seats')

# Average probabilities
axs[1, 0].bar(df['Party'], df['Average_Probs'])
axs[1, 0].set_title('Average Probabilities over 100 Runs')
axs[1, 0].set_xlabel('Party')
axs[1, 0].set_ylabel('Probability')

# Combined graph
axs[1, 1].bar(df['Party'], df['2018_Seats'], alpha=0.5, label='2018 Seats')
axs[1, 1].bar(df['Party'], df['Average_Seats'], alpha=0.5, label='Avg Seats')
axs[1, 1].set_title('Combined Graph')
axs[1, 1].set_xlabel('Party')
axs[1, 1].set_ylabel('Seats')
axs[1, 1].legend()

plt.tight_layout()
plt.show()
