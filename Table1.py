import pandas as pd
import numpy as np


age = pd.read_csv('data_table1_age.csv', index_col = 0)
df1 = pd.read_csv('data_table1_all.csv', index_col = 0)
patients = pd.read_csv('data_2800.csv', index_col = 0)
extra = pd.read_excel('demographic_not_in_thunderpacks.xlsx')
need = extra[['patientid', 'DateOfBirth', 'PatientRace', 'SexDSC']]
need = need.rename(columns = {'patientid': 'BDSPPatientID'})
df1 = df1[['BDSPPatientID', 'DateOfBirth', 'PatientRace', 'SexDSC']]

df = pd.concat([df1, need], ignore_index = True)





mgh = patients[patients['hospital']== 'MGH']
bidmc = patients[patients['hospital']== 'BIDMC']
mgh = mgh.rename(columns = {'patientid': 'BDSPPatientID'})
bidmc = bidmc.rename(columns = {'patientid': 'BDSPPatientID'})

mgh = mgh.drop_duplicates(subset='BDSPPatientID', keep='first')

mgh_dat = pd.merge(mgh, age[['BDSPPatientID', 'DateOfBirth']], on = 'BDSPPatientID', how='left')
mgh_dat['note_date'] = pd.to_datetime(mgh_dat['note_date'])
mgh_dat['DateOfBirth'] = pd.to_datetime(mgh_dat['DateOfBirth'])
mgh_dat['age'] = (mgh_dat['note_date'] - mgh_dat['DateOfBirth']).dt.days /365.25


bidmc = bidmc.drop_duplicates(subset='BDSPPatientID', keep='first')
bidmc_dat = pd.merge(bidmc, age[['BDSPPatientID', 'DateOfBirth']], on = 'BDSPPatientID', how='left')
bidmc_dat['note_date'] = pd.to_datetime(bidmc_dat['note_date'])
bidmc_dat['DateOfBirth'] = pd.to_datetime(bidmc_dat['DateOfBirth'])
bidmc_dat['age'] = (bidmc_dat['note_date'] - bidmc_dat['DateOfBirth']).dt.days /365.25

mgh_sex = pd.merge(mgh_dat, df[['BDSPPatientID', 'PatientRace', 'SexDSC']], on = 'BDSPPatientID', how='left')
mgh_sex = mgh_sex.drop_duplicates(subset='BDSPPatientID', keep='first')
mgh_sex['female_yes'] = np.where(mgh_sex['SexDSC'].str.lower().isin(['female', 'f']), 1, 0)
mgh_sex['white_yes'] = np.where(mgh_sex['PatientRace'].str.lower() == 'white', 1, 0)
mgh_sex['asian_yes'] = np.where(mgh_sex['PatientRace'].str.lower() == 'asian', 1, 0)
mgh_sex['black_yes'] = np.where(mgh_sex['PatientRace'].str.lower().isin(['black or african american']), 1, 0)

bidmc_sex = pd.merge(bidmc_dat, df[['BDSPPatientID', 'PatientRace', 'SexDSC']], on = 'BDSPPatientID', how='left')
bidmc_sex = bidmc_sex.drop_duplicates(subset='BDSPPatientID', keep='first')
bidmc_sex['female_yes'] = np.where(bidmc_sex['SexDSC'].str.lower().isin(['female', 'f']), 1, 0)
bidmc_sex['white_yes'] = np.where(bidmc_sex['PatientRace'].str.lower() == 'white', 1, 0)
bidmc_sex['asian_yes'] = np.where(bidmc_sex['PatientRace'].str.lower() == 'asian', 1, 0)
bidmc_sex['black_yes'] = np.where(bidmc_sex['PatientRace'].str.lower().isin(['black or african american']), 1, 0)
