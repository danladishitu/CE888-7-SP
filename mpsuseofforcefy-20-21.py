#!/usr/bin/env python
# coding: utf-8

# ## Metropolitan London Use of Force 
# 
# Each row set represents use of force by an officer on one subject. This row set is not a count of incident . Where more than one officer has used force on the same subject, this will be shown on separate rows of data. As such this will result in duplicate metadata. Further consideration needs to be taken when analysing this data to ensure correct conclusions are drawn. For example if 2 officers use force on an individual who happens to be male aged 18-34 this will be shown on 2 rows of data. Hence, if grouping the data, you would have a count of 2x males and 2x 18-34 which would be incorrect. 
# 
# 

# ### Link to the dataset:
# - [Metropolitan Use of Force] https://data.london.gov.uk/download/use-of-force/9d266ef1-7376-4eec-bb0d-dfbd2b1a591e/MPS%20Use%20of%20Force%20-%20FY20-21.xlsx

# In[66]:


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt 
import seaborn as sns

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline 


# ### About the dataset
# The dataset consist of **147,895 rows** and **271 columns**. We will not be able to train the model with all of this information because not all the features are needed. And in other not to slow down the performance of our system we will limit our samples to **25,000** records from the dataset. We will be doing some cleaning up and preprocessing on the data to suit the model requirements. 

# In[67]:


#Load the dataset that was gotten from the metropolitan London data store
police_df=pd.read_csv("mpsuseofforce20-21.csv" , encoding='cp1252')
police_df


# In[68]:


#We will use a sample of 25,000 records  
police_df= police_df.sample(25000, replace=True)


# ### Renaming the model
# It is important to know what our label represent and just to give a general overview of what each feature depicts. This is are the label we will be using all through the experiment. 
#  
# - Incident Location: Street/Highway - location
# - Incident Location: Open ground (e.g. park, car park, field) - openground
# - Incident Location: Sports or Event Stadia - stadium
# - Incident Location: Dwelling - ildwelling
# - Borough - borough
# - Impact Factor: Possesion of a weapon - ifpossessionweapon
# - Impact Factor: Alcohol - ifalcohol
# - Impact Factor: Drugs - drugs
# - Impact Factor: Mental Health - ifmentalhealth
# - SubjectAge - age
# - SubjectGender - gender
# - Ethnicity - race
# - MentalDisability - mentallydisable

# In[69]:


#Lets rename the columns we don't want to use a cumbersome label for our features 
police_df= police_df.rename(columns = {'Incident Location: Street/Highway': 'location', 'Incident Location: Public Transport': 'publictransport',
                                       'Incident Location: Retail Premises':'retailpremises',
                                       'Incident Location: Open ground (e.g. park, car park, field)': 'openground',
                                       'Incident Location: Licensed Premises': 'licensedpremises',
                                       'Incident Location: Sports or Event Stadia':'stadium',
                                       'Incident Location: Hospital/A&E (non-mental-health setting)':'nonmentalhealthsetting',
                                       'Incident Location: Mental Health Setting' :'mentalhealthsetting',
                                       'Incident Location: Police vehicle with prisoner handling cage':'policevehicle',
                                       'Incident Location: Police vehicle without prisoner handling cage': 'policevehiclewthprisoner',
                                       'Incident Location: Dwelling':'ildwelling',
                                       'Incident Location: Police station (excluding custody block)': 'ilpolicestation',
                                       'Incident Location: Custody Block':'ilcustodyblock',
                                      'Incident Location: Ambulance' :'ilambulance',
                                      'Incident Location: Other':'ilother',
                                      'Borough':'borough',
                                      'PrimaryConduct':'primaryconduct',
                                      'AssaultedBySubject':'assaultedbysubject',
                                      'Impact Factor: Possesion of a weapon':'ifpossessionweapon',
                                      'Impact Factor: Alcohol':'ifalcohol',
                                      'Impact Factor: Drugs' :'ifdrug',
                                      'Impact Factor: Mental Health':'ifmentalhealth',
                                      'Impact Factor: Prior Knowledge':'ifpriorknowledge',
                                      'Impact Factor: Size/Gender/Build':'ifbysizeandgender',
                                      'Impact Factor: Acute Behavioural Disorder':'ifbehaviorsdisorder',
                                      'Impact Factor: Other':'ifother',
                                      'Reason for Force: Protect self':'rfprotectself',
                                      'Reason for Force: Protect Public':'rfprotectpublic',
                                      'Reason for Force: Protect Subject':'rfprotectsubject',
                                      'Reason for Force: Protect Other Officers':'rfprotectofficer',
                                      'Reason for Force: Prevent Offence':'rfpreventoffence',
                                      'Reason for Force: Secure Evidence' :'rfsecureevidence',
                                      'Reason for Force: Effect Search' :'rfeffectsearch',
                                      'Reason for Force: Effect Arrest' :'rfeffectarrest',
                                      'Reason for Force: Method of Entry':'rfmethodofentry',
                                      'Reason for Force: Remove Handcuffs':'rfremovehandcuffs',
                                      'Reason for Force: Prevent Harm':'rfpreventharm',
                                      'Reason for Force: Prevent Escape':'rfpreventescape',
                                      'Reason for Force: Other':'rfothers',
                                      'MainDuty':'mainduty',
                                      'SingleCrewed':'singlecrewed',
                                      'TrainedCED':'trainedced',
                                      'CarryingCED':'carryingced',
                                      'Tactic 1':'tacticone',
                                      'Effective 1':'effectiveone',
                                      'Tactic 2':'tactictwo',
                                      'Effective 2':'effectivetwo',
                                      'Tactic 3':'tacticthree',
                                      'Effective 3':'effectivethree',
                                      'Firearms Aimed':'fraimed',
                                      'Firearms Fired':'frfired',
                                      'SubjectAge':'age',
                                      'SubjectGender':'gender',
                                      'SuBlackjectEthnicity':'race',
                                      'PhysicalDisability':'physicaldisable',
                                      'MentalDisability':'mentallydisable',
                                      'StaffInjured':'policeinjured',
                                      'SubjectInjured':'subjectinjured',
                                      'Outcome: Made off/escaped':'escaped',
                                      'Outcome: Arrested':'arrested'}, inplace = False)
police_df.head(5)


# In[70]:


#We will drop some features that are not important for our models
police_df=police_df.drop(['IncidentDate','IncidentTime','publictransport','ilambulance','retailpremises','nonmentalhealthsetting','mentalhealthsetting','policevehicle','policevehiclewthprisoner','ilpolicestation',
                         'ilcustodyblock','policeinjured', 'subjectinjured','assaultedbysubject','primaryconduct','ifbehaviorsdisorder','rfprotectself','rfprotectpublic','licensedpremises','rfprotectofficer','rfpreventharm','rfpreventescape','escaped','arrested','fraimed','ilother','ThreatenedWithWeapon','AssaultedWithWeapon','ifpriorknowledge','ifbysizeandgender','ifother','Impact Factor: Crowd','rfeffectsearch','rfmethodofentry','rfothers','mainduty',
                         'singlecrewed','trainedced','rfpreventoffence','rfsecureevidence','rfeffectarrest','rfprotectsubject','carryingced','tacticone','effectiveone','tactictwo','effectivetwo','tacticthree','effectivethree','Tactic 4','Effective 4','Tactic 5','Effective 5','Tactic 6',
                         'Effective 6','Tactic 7','Effective 7','Tactic 8','Effective 8','Tactic 9','Effective 9','Tactic 10','Effective 10','Effective 11','Tactic 11','Tactic 12','Effective 12','Tactic 13','rfremovehandcuffs',
                         'Effective 13','Tactic 14','Effective 14','Tactic 15','Effective 15','Tactic 16','Effective 16','Tactic 17','Effective 17','Tactic 18','Effective 18','Tactic 19','Effective 19','Tactic 20',
                         'Tactic Effective 20','CED Used','CED Device Serial No','CED Drawn', 'CED Aimed','CED Arced','CED Red-Dotted','CED Drive Stun','CED Drive Stun Repeat Application', 'CED Angle Drive Stun','CED Fired',
                         'CED Fired Cartridge Number','CED Fired 5 Secs Cycle Interrupted','CED Fired Repeat Cycle Same Cartridge','CED Fired Total Number Of Cycles','CED Fired Cycle Extended Beyond 5 Secs',
                         'CED Fired Miss With One Probe','CED Fired Miss With Both Probes','CED Front 1','CED Front 2','CED Front 3','CED Front 4','CED Front 5','CED Front 6','CED Front 7','CED Front 8','CED Front 9',
                         'CED Front 10','CED Front 11','CED Front 12','CED Front 13','CED Front 14','CED Front 15','CED Back A','CED Back B','CED Back C','CED Back D', 'CED Back E','CED Back F','CED Back G','CED Back H','CED Back J',
                         'CED Back K','CED2 Drawn','CED2 Aimed','CED2 ArCED2','CED2 Red-Dotted','CED2 Drive Stun','CED2 Drive Stun Repeat Application','CED2 Angle Drive Stun','CED2 Fired','CED2 Fired Cartridge Number',
                         'CED2 Fired 5 Secs Cycle Interrupted','CED2 Fired Repeat Cycle Same Cartridge','CED2 Fired Total Number Of Cycles','CED2 Fired Cycle Extended Beyond 5 Secs','CED2 Fired Miss With One Probe',
                         'CED2 Fired Miss With Both Probes','CED2 Front 1','CED2 Front 2','CED2 Front 3','CED2 Front 4','CED2 Front 5','CED2 Front 6','CED2 Front 7','CED2 Front 8','CED2 Front 9','CED2 Front 10','CED2 Front 11',
                         'CED2 Front 12','CED2 Front 13','CED2 Front 14','CED2 Front 15','CED2 Back A','CED2 Back B','CED2 Back C','CED2 Back D','CED2 Back E','CED2 Back F','CED2 Back G','CED2 Back H','CED2 Back J','CED2 Back K','CED3 Drawn',
                         'CED3 Aimed','CED3 ArCED3','CED3 Red-Dotted', 'CED3 Drive Stun','CED3 Drive Stun Repeat Application','CED3 Angle Drive Stun','CED3 Fired','CED3 Fired Cartridge Number','CED3 Fired 5 Secs Cycle Interrupted',
                         'CED3 Fired Repeat Cycle Same Cartridge','CED3 Fired Total Number Of Cycles','CED3 Fired Cycle Extended Beyond 5 Secs','CED3 Fired Miss With One Probe','CED3 Fired Miss With Both Probes','CED3 Front 1','CED3 Front 2',
                         'CED3 Front 3','CED3 Front 4','CED3 Front 5','CED3 Front 6','CED3 Front 7','CED3 Front 8','CED3 Front 9','CED3 Front 10','CED3 Front 11','CED3 Front 12','CED3 Front 13','CED3 Front 14','CED3 Front 15','CED3 Back A',
                         'CED3 Back B','CED3 Back C','CED3 Back D','CED3 Back E','CED3 Back F','CED3 Back G','CED3 Back H','CED3 Back J','CED3 Back K','CED4 Drawn','CED4 Aimed','CED4 ArCED4','CED4 Red-Dotted','CED4 Drive Stun',
                         'CED4 Drive Stun Repeat Application','CED4 Angle Drive Stun','CED4 Fired','CED4 Fired Cartridge Number','CED4 Fired 5 Secs Cycle Interrupted','CED4 Fired Repeat Cycle Same Cartridge','CED4 Fired Total Number Of Cycles',
                         'CED4 Fired Cycle Extended Beyond 5 Secs','CED4 Fired Miss With One Probe','CED4 Fired Miss With Both Probes','CED4 Front 1','CED4 Front 2','CED4 Front 3','CED4 Front 4','CED4 Front 5','CED4 Front 6','CED4 Front 7','CED4 Front 8',
                         'CED4 Front 9','CED4 Front 10','CED4 Front 11','CED4 Front 12','CED4 Front 13','CED4 Front 14','CED4 Front 15','CED4 Back A','CED4 Back B','CED4 Back C','CED4 Back D','CED4 Back E','CED4 Back F','CED4 Back G','CED4 Back H',
                         'CED4 Back J','CED4 Back K','frfired','physicaldisable','StaffInjuryIntentional','StaffInjuryLevel','StaffMedProvided','SubjectNatureOfInjury','SubjectMedOffered','SubjectMedProvided',
                         'Outcome: Hospitalised','Outcome: Detained - Mental Health Act','Outcome: Fatality','Outcome: Other'], axis=1)
police_df


# In[71]:


#Let's visualize to see the cities where the crimes are committed.  
plt.figure(figsize=(10,10))
city = [i for i in police_df['borough']]
city_dict = {x:city.count(x) for x in city}
plt.pie(city_dict.values(), labels=city_dict.keys(),autopct='%.2f',startangle=300)
plt.title('Metropolitan Cities');


# In[72]:


#Lets visualize the race 
ax = sns.countplot(x = police_df['race'])
plt.figure(figsize=(80, 40))
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
ax.set_xlabel('Race') 
ax.set_ylabel('Range')
plt.tight_layout()
plt.show()


# In[73]:


#Drop all Nan (Note a number)
print("Number of rows before dropping NaNs: %d" % len(police_df))
police_df = police_df.dropna()
print("Number of rows after dropping NaNs: %d" % len(police_df))


# In[74]:


#Import library to perform over sampling using Synthetic Minority Over-sampling Techniques
import imblearn
print(imblearn.__version__)
from imblearn.over_sampling import SMOTE


# In[76]:


#Import standard scaler preprocessor to store all our numerical value in an array format 
from sklearn.preprocessing import LabelEncoder
police_df[["borough"]] = police_df[["borough"]].apply(LabelEncoder().fit_transform)
police_df[["location"]] = police_df[["location"]].apply(LabelEncoder().fit_transform)
police_df[["openground"]] = police_df[["openground"]].apply(LabelEncoder().fit_transform)
police_df[["race"]] = police_df[["race"]].apply(LabelEncoder().fit_transform)
police_df[["age"]] = police_df[["age"]].apply(LabelEncoder().fit_transform)
police_df[["gender"]] = police_df[["gender"]].apply(LabelEncoder().fit_transform)
police_df[["mentallydisable"]] = police_df[["mentallydisable"]].apply(LabelEncoder().fit_transform)
police_df[["stadium"]] = police_df[["stadium"]].apply(LabelEncoder().fit_transform)
police_df[["ildwelling"]] = police_df[["ildwelling"]].apply(LabelEncoder().fit_transform)
police_df[["ifalcohol"]] = police_df[["ifalcohol"]].apply(LabelEncoder().fit_transform)
police_df[["ifdrug"]] = police_df[["ifdrug"]].apply(LabelEncoder().fit_transform)
police_df[["ifmentalhealth"]] = police_df[["ifmentalhealth"]].apply(LabelEncoder().fit_transform)
police_df.head(15)


# In[77]:


#Convert a string variable to a categorical one then fit X,Y sampling
X = police_df.drop('ifpossessionweapon', axis=1)

Y = police_df['ifpossessionweapon']
Y.head()


# In[ ]:




