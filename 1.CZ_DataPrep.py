# CZ_DataPrep
# Extract climate data from ClimateNA ascii files and aggregate data for multivariate analyses

# Import necessary libraries
import pandas as pd
import numpy as np
import os

# Define file and folder paths using raw strings (r'') to avoid escape character issues
DomainFile = r"D:\inspires\data\ClimateNA\ClimateNA_Reference\ClimateNA_mask.csv"
IDfile = r"D:\inspires\data\ClimateNA\ClimateNA_Reference\ClimateNA_ID.csv"
dataDir = r"D:\inspires\data\ClimateNA\1961-1990\\"
ID_subselectSave = r"D:\inspires\data\ClimateNA\1961-1990.csv"

# Read the ClimateNA ID file into a pandas DataFrame and rename the first column as 'ID'
all_data = pd.read_csv(IDfile, delimiter=',', header=None)
all_data=all_data.rename(columns={0: 'ID'})

# Retrieve the list of ASC files in the data folder
data_files = [file for file in os.listdir(dataDir) if file.endswith(".asc")]

# Define the row and column indices for the data sub-selection and the maximum number of lines to read at a time
toprow = 3302
bottomrow = 4498
firstcol = 5557
lastcol = 6900
maxlines = 50.

# Define the header length for the ASC files
asc_header = 6

# Calculate the number of iterations needed to read all the rows in the data sub-selection
iterations = int(round((bottomrow-toprow)/maxlines))

# Loop through each file in the list of ASC files
for file in data_files:
    # Print the name of the file being processed
    print(file[:-4])
    
    # Initialize an empty numpy array to hold the sub-selected data
    data_subselect = np.array([])
    
    # Loop through each iteration and read in a block of data from the ASC file and domain mask file
    for i in range(iterations):
        start_row = toprow + maxlines * i
        
        # Read in the domain mask data and flatten it into a 1D numpy array
        df_mask = pd.read_csv(DomainFile, delimiter=',', skiprows=toprow, nrows=maxlines, usecols=range(firstcol,lastcol), header=None)
        ndf_mask = df_mask.to_numpy().flatten()

        # Read in the ASC file data and flatten it into a 1D numpy array
        df_data = pd.read_csv(dataDir+file, delimiter=' ', skiprows=start_row+asc_header, nrows=maxlines, usecols=range(firstcol,lastcol),header=None)
        ndf_data = df_data.to_numpy().flatten()

        # Select only the values in the data array that correspond to cells in the domain mask with a value of 1
        data_subselect = np.append(data_subselect, ndf_data[ndf_mask==1])

    # Try to add the sub-selected data to the all_data DataFrame with the name of the file as the column label
    try:
        all_data['%s' %file[:-4]] = data_subselect
    except:
        # If an error occurs, enter pdb.set_trace() to enter debug mode and investigate the issue
        import pdb; pdb.set_trace()

# Write the sub-selected data to a CSV file
all_data.to_csv(ID_subselectSave,index=False)

