import pandas as pd
import numpy as np


def select_Columns_fromCSV(filepath, columns, keyColumn, surveyID):
	"""
	select desired columns from one CSV file.
	then select all rows that has the surveyID.

	Parameters:
	----------------
	filepath: str
			The path of CSV file
	columns: lst
			A list of strings of desired columns, including the keyColumn
	keyColumn: str
			The column for indexing or satisfing conditions
	surveyID: int
			Desired ID value in a key column
	Return:
	----------------
	data: dataframe
			Pandas dataframe
	"""
	dataCSV = pd.read_csv(filepath, usecols=columns)
	dataID = dataCSV[dataCSV[keyColumn] == surveyID]
	return dataID

def replace_values(data, column, oldValues, newValues):
	"""
	replace old values with new values in a column in the datafame

	Parameters:
	----------------
	data:   dataframe
			Pandas dataframe
	column: str
			A desired column
	oldValues: lst
			A list of all old values to be changed
	newValues: lst
			A list of all desired new values
	Return:
	----------------
	data: dataframe
			Pandas dataframe
	"""
	for i in range(len(oldValues)):
        # the column needs to be pure characters without quotations
		data[column] = np.where((data.column == oldValues[i]), newValues[i], data.column)
	return data
