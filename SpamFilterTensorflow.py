import pandas
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

FilePath = 'spam.csv'
processedFilePath = 'Processed.csv'

"""
Clenup the the dataset and delete all of the non utf-8 character. Write the after processed file into Processed.csv
 """
with open(FilePath, 'rb') as csv_in:
    with open(processedFilePath, "w", encoding="utf-8") as csv_temp:
        for line in csv_in:
            if not line:
                break
            else:
                line = line.decode("utf-8", "ignore")
                csv_temp.write(str(line).rstrip() + '\n')
# Load the CSV file into a pandas dataframe
data = pd.read_csv(processedFilePath, encoding='utf-8')


# Assign label and data
messages = data['v2'].values
labels = data['v1'].values

CATEGORIES = {
  'Spam': ['spam'],
  'Harm': ['ham']
}

