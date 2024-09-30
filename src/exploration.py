import pandas as pd

# Load the training data from the CSV file Some data inconsistency which have been fixed manually  in line 102 line 602 line 
data = pd.read_csv("training_data_en.csv", sep=';')

# remove duplicates and reset index
data2=data[['Text', 'Question', 'Answer']].copy()
data2.drop_duplicates(inplace=True)
data2.reset_index(drop=True, inplace=True)

# Print the data size
print("Sie of the data:")
print(len(data2))

# Print the column names
print("Column names:")
print(data.columns)

# Create a new column to indicate if the answer is part of the text
data['is_answer_in_text'] = data.apply(lambda row: row['Answer'] not in row['Text'], axis=1)

# Print the number of answers that are not part of the text
print(f"Number of answers not in text: {sum(data['is_answer_in_text'])}")

# Get the indexes of the inconsistent data
inconsistent_data_indexes = data[data['is_answer_in_text']].index.values

# Print the inconsistent data
print("Inconsistent data:")
for idx in inconsistent_data_indexes:
    print(f"{data.iloc[idx]['ID']};{data.iloc[idx]['Text']};{data.iloc[idx]['Question']};{data.iloc[idx]['Answer']}")

# Note: The following comments are manually added to highlight the inconsistencies
# ...
print(""" Note: The following comments are manually added to highlight the inconsistencies
# 3373; ... (missing "Life on land")
# 5364.3; ... (missing - "in one off")
# 4039.a; ... (rewritten answer)
# 6014.b; ... (extra space in "re serve")
# 2564; ... (no text in this example)
# 2587; ... (quotes not included in answer)
# 3681.a; ... (maybe extra space)
# 5221; ... (same as 2587, quotes not included in answer)
# 5269.3.b; ... (extra space in "re-financing")
# 3146.b; ... (capital "T")
# 3965; ... (extra "Remuneration Policy")
# 4047; ... (extra space in "currency-denominated")""")


## finding the stats for the Text


data['word_count'] = data['Text'].apply(lambda x: len(x.split()))
stats = data['word_count'].describe()
max_length = stats['max']
min_length = stats['min']
avg_length = stats['mean']

print(f"Text Max word count: {max_length}")
print(f"Text Min word count: {min_length}")
print(f"Text Average word count: {avg_length}")

## finding the stats for the Question


data['word_count'] = data['Question'].apply(lambda x: len(x.split()))
stats = data['word_count'].describe()
max_length = stats['max']
min_length = stats['min']
avg_length = stats['mean']

print(f"Question Max word count: {max_length}")
print(f"Question Min word count: {min_length}")
print(f"Question Average word count: {avg_length}")


## finding the stats for the Answer

data['word_count'] = data['Answer'].apply(lambda x: len(x.split()))
stats = data['word_count'].describe()
max_length = stats['max']
min_length = stats['min']
avg_length = stats['mean']

print(f"Answer Max word count: {max_length}")
print(f"Answer Min word count: {min_length}")
print(f"Answer Average word count: {avg_length}")

Question=''
Text=''
    
f"""
Extract the relevant text that answers the following question:

'''
{Question}
'''

from the given text:

'''
{Text}
'''


Provide the exact phrase or sentence from the original text that directly answers the question."
"""
"Note:does it make a difference if we give the text first then the question?"