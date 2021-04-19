#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


electric=pd.read_csv('/Users/ads2137/Downloads/station_elec_equipments_mv.csv')
#read in data on where EV charging stations exist in the US
#data from https://maps.nrel.gov/transatlas
electric.head()


# In[3]:


gas_usage=pd.read_csv('/Users/ads2137/Downloads/state_gas_use.csv')
#read in highway use of gasoline by state
#data from https://www.fhwa.dot.gov/policyinformation/statistics/2018/mf226.cfm
gas_usage.head()


# In[4]:


public_E_states =pd.DataFrame()
public_E_states['state']=electric['st_prv_code'][(electric['status_code']=='AVBL') & (electric['groups_with_access_code']=='Public')]
#filter for publically availably EV charging stations only


# In[5]:


public_E_states['code']=electric['status_code'][(electric['status_code']=='AVBL') & (electric['groups_with_access_code']=='Public')]
public_E_states['access']=electric['groups_with_access_code'][(electric['status_code']=='AVBL') & (electric['groups_with_access_code']=='Public')]
#print(public_E_states.info())
#add status code and access to check filtering


# In[6]:


#print(public_E_states.head(25))


# In[7]:


#print(public_E_states.value_counts())
# see how states rank in publically available EV charging stations


# In[8]:


#print(gas_usage.head(25))


# In[9]:


# United States of America Python Dictionary to translate States,
# Districts & Territories to Two-Letter codes and vice versa.
#
# https://gist.github.com/rogerallen/1583593
#
# Dedicated to the public domain.  To the extent possible under law,
# Roger Allen has waived all copyright and related or neighboring
# rights to this code.
us_state_abbrev = {
    'Alabama': 'AL',
    'Alaska': 'AK',
    'American Samoa': 'AS',
    'Arizona': 'AZ',
    'Arkansas': 'AR',
    'California': 'CA',
    'Colorado': 'CO',
    'Connecticut': 'CT',
    'Delaware': 'DE',
    'District of Columbia': 'DC',
    'Florida': 'FL',
    'Georgia': 'GA',
    'Guam': 'GU',
    'Hawaii': 'HI',
    'Idaho': 'ID',
    'Illinois': 'IL',
    'Indiana': 'IN',
    'Iowa': 'IA',
    'Kansas': 'KS',
    'Kentucky': 'KY',
    'Louisiana': 'LA',
    'Maine': 'ME',
    'Maryland': 'MD',
    'Massachusetts': 'MA',
    'Michigan': 'MI',
    'Minnesota': 'MN',
    'Mississippi': 'MS',
    'Missouri': 'MO',
    'Montana': 'MT',
    'Nebraska': 'NE',
    'Nevada': 'NV',
    'New Hampshire': 'NH',
    'New Jersey': 'NJ',
    'New Mexico': 'NM',
    'New York': 'NY',
    'North Carolina': 'NC',
    'North Dakota': 'ND',
    'Northern Mariana Islands':'MP',
    'Ohio': 'OH',
    'Oklahoma': 'OK',
    'Oregon': 'OR',
    'Pennsylvania': 'PA',
    'Puerto Rico': 'PR',
    'Rhode Island': 'RI',
    'South Carolina': 'SC',
    'South Dakota': 'SD',
    'Tennessee': 'TN',
    'Texas': 'TX',
    'Utah': 'UT',
    'Vermont': 'VT',
    'Virgin Islands': 'VI',
    'Virginia': 'VA',
    'Washington': 'WA',
    'West Virginia': 'WV',
    'Wisconsin': 'WI',
    'Wyoming': 'WY'
}

# thank you to @kinghelix and @trevormarburger for this idea
abbrev_us_state = dict(map(reversed, us_state_abbrev.items()))
# dictionary to convert state abbrevations to state names


# In[10]:


public_E_states['state'].replace(abbrev_us_state,inplace=True)
#print(public_E_states.value_counts())
#convert to state names for consistency


# In[11]:


gas_usage_2000=pd.DataFrame()
gas_usage_2000=gas_usage[['STATE','2000','2001','2002','2003','2004','2005','2006','2007','2008','2009','2010','2011','2012','2013','2014','2015','2016','2017','2018']]
#filtered dataframe for only more recent years


# In[12]:


gas_usage_2000=gas_usage_2000.drop([51,52,53,54])
#print(gas_usage_2000)
#drop nans and Puerto Rico


# In[13]:


to_fix=['2000','2001','2002','2003','2004','2005','2006','2007','2008','2009','2010','2011','2012','2013','2014','2015','2016','2017','2018']

for col in to_fix:
    gas_usage_2000[col] = gas_usage_2000[col].str.replace(',','').astype(float)
#clean gas usage and convert to float


# In[14]:


train_gas=gas_usage_2000.iloc[0:26]
test_gas=gas_usage_2000.iloc[26:]
#print(test_gas)



# In[15]:


public_E_states.rename(columns={"state": "STATE"},inplace=True)


# In[16]:


gas_usage_2000["STATE"].replace('Dist. of Col.',"District of Columbia",inplace=True)
#make DC name consistent across both current dataframes


# In[17]:


state_counts=public_E_states['STATE'].value_counts()
state_counts=state_counts.drop('Puerto Rico')
#get counts of EV charging station per state
#drop puerto rico from EV charging state counts for now
#print(state_counts)


# In[18]:


state_counts=list(state_counts.sort_index())
#print(state_counts)
#get list of state counts of EV charging sorted albhabetically by state for consistency with gas usage dataframe


# In[19]:


gas_usage_2000['e_counts']=state_counts
#print(gas_usage_2000.head())
#add EV charging station counts to recent gas_usage data frame for convenience of understanding gas vs EV relationship


# In[20]:


print(gas_usage_2000.sort_values(by='e_counts',ascending=False))
#look at data sorted by amount of EV charing stations


# In[21]:


fig,ax=plt.subplots(1,1,figsize=(10,10))

ax.scatter(gas_usage_2000['e_counts'],gas_usage_2000['2018'],label='Other States')
ax.scatter(gas_usage_2000['e_counts'][gas_usage_2000['STATE']=='Texas'],gas_usage_2000['2018'][gas_usage_2000['STATE']=='Texas'],label='Texas')
ax.scatter(gas_usage_2000['e_counts'][gas_usage_2000['STATE']=='California'],gas_usage_2000['2018'][gas_usage_2000['STATE']=='California'],label='California')


plt.annotate("TX",[gas_usage_2000['e_counts'][gas_usage_2000['STATE']=='Texas'],gas_usage_2000['2018'][gas_usage_2000['STATE']=='Texas']],size=16)
plt.annotate("CA",[gas_usage_2000['e_counts'][gas_usage_2000['STATE']=='California'],gas_usage_2000['2018'][gas_usage_2000['STATE']=='California']],size=16)

ax.tick_params(axis='x', size= 14)
ax.tick_params(axis='y', size= 14)

ax.set_xlabel('Public EV Charging Stations in The State',size=16)
ax.set_ylabel('Gas Use in 2018 (Thousands of Gallons)',size=16)

ax.set_title('Texas has High Highway Use of Gasoline',size=20)
ax.legend(prop={'size': 14},loc='lower right')
plt.show()

#California is high in charging stations and gas use
#Texas is also high in gas use by does not have high EV charging 
#gas use and EV charging are generally correlated in other (lower pop.) states



# In[22]:


#print(gas_usage_2000.sort_values(by='2018',ascending=False))
#sort by highest gas usage in 2018


# In[23]:


gas_and_e=gas_usage_2000
#rename plot that contains both EV and gas info


# In[24]:


gas_usage_2000=gas_usage_2000.drop('e_counts',axis=1)
#print(gas_usage_2000.head())
#regenerate gas only df for plotting


# In[25]:


fig,ax=plt.subplots(1,1,figsize=(15,7.5))

ax.plot(gas_usage_2000.set_index('STATE').T['California'],label='California')
ax.plot(gas_usage_2000.set_index('STATE').T['Texas'],label='Texas')
ax.tick_params(axis='x', size= 14)
ax.tick_params(axis='y', size= 14)

ax.set_xlabel('Gas Use in 2018 (Thousands of Gallons)',size=16)
ax.set_ylabel('Time (Years)',size=16)

ax.set_title('Texas Highway Use of Gasoline has Increased in Recent Years',size=20)
ax.legend(prop={'size': 14})
plt.show()


#Texas has been increasing in gas usage over the last decade while California has not.


# In[26]:


turbines=pd.read_csv('/Users/ads2137/Downloads/uswtdbCSV/uswtdb_v4_0_20210409.csv')
#read in wind turbine data
#data from https://eerscmap.usgs.gov/uswtdb/data/
#Citation:
#Hoen, B.D., Diffendorfer, J.E., Rand, J.T., Kramer, L.A., Garrity, C.P., and Hunt, H.E., 2018, 
#United States Wind Turbine Database (V4.0, (April 9, 2021): U.S. Geological Survey, American Clean Power Association, and Lawrence Berkeley National Laboratory data release, https://doi.org/10.5066/F7TX3DN0.


# In[27]:


turbines.head()


# In[28]:


turb_counts=turbines['t_state'].replace(abbrev_us_state,inplace=True)
#change state abbreviations to names


# In[29]:


#print(turbines.head())


# In[30]:


turb_counts=turbines['t_state'].value_counts()
#print(turb_counts)
#get wind turbine counts by state


# In[31]:


turb_counts=turb_counts.sort_index()
#print(turb_counts)
#sort by state name, not all states included in this data


# In[32]:


print(len(turb_counts))
#confirming not all states included


# In[33]:


print(turb_counts.index)
#confirming not all states included



# In[34]:


turb_counts_new=pd.DataFrame()
turb_counts_new['STATE']=list(turb_counts.index)
#print(turb_counts_new)
#making dataframe of turbine counts sorted by state name


# In[35]:


turb_counts_new['t_counts']=list(turb_counts)
#added turbmine counts


# In[36]:


gas_E_T=pd.merge(gas_and_e, turb_counts_new, on="STATE")
#print(gas_E_T)
#merging to get data that has information on gas, EV charging, and turbines on a subset of states


# In[37]:


fig,ax=plt.subplots(1,1,figsize=(10,10))

ax.scatter(gas_E_T['t_counts'],gas_E_T['2018'],label='Other States')
ax.scatter(gas_E_T['t_counts'][gas_E_T['STATE']=='Texas'],gas_E_T['2018'][gas_E_T['STATE']=='Texas'],label='Texas')
ax.scatter(gas_E_T['t_counts'][gas_E_T['STATE']=='California'],gas_E_T['2018'][gas_E_T['STATE']=='California'],label='California')


plt.annotate("TX",[gas_E_T['t_counts'][gas_E_T['STATE']=='Texas'],gas_E_T['2018'][gas_E_T['STATE']=='Texas']],size=16)
plt.annotate("CA",[gas_E_T['t_counts'][gas_E_T['STATE']=='California'],gas_E_T['2018'][gas_E_T['STATE']=='California']],size=16)

ax.tick_params(axis='x', size= 14)
ax.tick_params(axis='y', size= 14)

ax.set_xlabel('Number of Wind Turbines in the State',size=16)
ax.set_ylabel('Gas Use in 2018 (Thousands of Gallons)',size=16)

ax.set_title('TX Exceeds CA in Wind Turbines',size=20)
ax.legend(prop={'size': 14},loc='lower right')
plt.show()
#Texas and California switch places when it comes to high wind turbines


# In[38]:


fig,ax=plt.subplots(1,1,figsize=(10,10))

ax.scatter(gas_E_T['t_counts'],gas_E_T['e_counts'],label='Other States')
ax.scatter(gas_E_T['t_counts'][gas_E_T['STATE']=='Texas'],gas_E_T['e_counts'][gas_E_T['STATE']=='Texas'],label='Texas')
ax.scatter(gas_E_T['t_counts'][gas_E_T['STATE']=='California'],gas_E_T['e_counts'][gas_E_T['STATE']=='California'],label='California')


plt.annotate("TX",[gas_E_T['t_counts'][gas_E_T['STATE']=='Texas'],gas_E_T['e_counts'][gas_E_T['STATE']=='Texas']],size=16)
plt.annotate("CA",[gas_E_T['t_counts'][gas_E_T['STATE']=='California'],gas_E_T['e_counts'][gas_E_T['STATE']=='California']],size=16)

ax.tick_params(axis='x', size= 14)
ax.tick_params(axis='y', size= 14)

ax.set_xlabel('Number of Wind Turbines in the State',size=16)
ax.set_ylabel('Public EV Charging Stations in The State',size=16)

ax.set_title('TX Green Energy Could Potentially Power New EV Charging Stations',size=20)
ax.legend(prop={'size': 14},loc='lower right')
plt.show()
#plotting number of EV charging stations versus turbine number



# In[39]:


texas_turb_locs=pd.DataFrame()
texas_turb_locs['xlong']=turbines['xlong'][turbines['t_state']=='Texas']
#get texas turbine longitudes


# In[40]:


texas_turb_locs['ylat']=turbines['ylat'][turbines['t_state']=='Texas']
#get texas turbine latitudes
#print(texas_turb_locs)


# In[41]:


#print(electric.head())


# In[42]:


texas_e_locs=pd.DataFrame()
texas_e_locs['xlong']=electric['longitude'][(electric['st_prv_code']=='TX') & (electric['status_code']=='AVBL') & (electric['groups_with_access_code']=='Public')]
#get publically available EV chargin station longitudes


# In[43]:


texas_e_locs['ylat']=electric['latitude'][(electric['st_prv_code']=='TX') & (electric['status_code']=='AVBL') & (electric['groups_with_access_code']=='Public')]
#get publically available EV chargin station latitudes
#print(texas_e_locs)



# In[44]:


fig,ax=plt.subplots(1,1,figsize=(10,10))
ax.scatter(texas_e_locs['xlong'],texas_e_locs['ylat'],c='#91bfdb',label='Existing EV Charging Stations')
ax.scatter(texas_turb_locs['xlong'],texas_turb_locs['ylat'], c='#fc8d59',label='Existing Wind Turbines')


ax.legend(prop={'size': 14})
ax.tick_params(axis='x', size= 14)
ax.tick_params(axis='y', size= 14)

ax.set_xlabel('Longitude (deg)',size=16)
ax.set_ylabel('Longitude (deg)',size=16)

ax.set_title('Existing EV Charging Stations Do Not Cluster near Wind Turbines',size=20)

plt.show()
#plotting locations of turbines and EV stations in texas


# In[45]:


texas_e_locs['label']='EVC'
texas_turb_locs['label']='turb'
texas_locs=pd.concat([texas_e_locs,texas_turb_locs])
#combine into one dataframe with labels to distinguish the two



# In[46]:


truckstops=pd.read_csv('/Users/ads2137/Downloads/Truck_Stop_Parking.csv')

#read in truck stop data
#data from https://data-usdot.opendata.arcgis.com/datasets/truck-stop-parking/data?geometry=-156.331%2C24.870%2C-35.042%2C49.055
truckstops.head()


# In[47]:


texas_truckstop_locs=pd.DataFrame()
texas_truckstop_locs['xlong']=truckstops['Longitude'][truckstops['State']=='TX']
texas_truckstop_locs['ylat']=truckstops['Latitude'][truckstops['State']=='TX']

#get texas truck stop locations
texas_locs=pd.concat([texas_e_locs,texas_turb_locs])
texas_truckstop_locs['label']='truckstop'
texas_locs=pd.concat([texas_locs,texas_truckstop_locs])
#add to overall location dataframe for convenience


# In[48]:



fig,ax=plt.subplots(1,1,figsize=(10,10))
ax.scatter(texas_e_locs['xlong'],texas_e_locs['ylat'],c='#91bfdb',label='Existing EV Charging Stations')
ax.scatter(texas_turb_locs['xlong'],texas_turb_locs['ylat'], c='#fc8d59',label='Existing Wind Turbines')
ax.scatter(texas_truckstop_locs['xlong'],texas_truckstop_locs['ylat'],c='#d73027',label='Existing Truck Stops')

ax.legend(prop={'size': 14})
ax.tick_params(axis='x', size= 14)
ax.tick_params(axis='y', size= 14)

ax.set_xlabel('Longitude (deg)',size=16)
ax.set_ylabel('Longitude (deg)',size=16)

ax.set_title('Truck Stops are more Disperse than Wind Turbines',size=20)
plt.show()


#plot data again truck stop locations


# In[49]:


#current EV stations are condensed around major cities
#perform density based clustering to try to pull out dense clusters 
X=texas_e_locs[['xlong', 'ylat']]

from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

# Compute DBSCAN
db = DBSCAN().fit(X)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

texas_e_locs['cluster'] = db.labels_



# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)



print('Estimated number of clusters: %d' % n_clusters_)
print('Estimated number of noise points: %d' % n_noise_)


# # Plot result

fig,ax=plt.subplots(1,1,figsize=(10,10))
# Black removed and is used for noise instead.
unique_labels = set(labels)
colors = [plt.cm.Spectral(each)
          for each in np.linspace(0, 1, len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]
    class_member_mask = (labels == k)
    


    xy = X[class_member_mask & core_samples_mask]
    #print(xy.iloc[:, 0])
    #print(xy[:, 0])
    ax.plot(xy.iloc[:, 0], xy.iloc[:, 1], 'o', markerfacecolor=tuple(col),markeredgecolor='k', markersize=14)

    xy = X[class_member_mask & ~core_samples_mask]
    ax.plot(xy.iloc[:, 0], xy.iloc[:, 1], 'o', markerfacecolor=tuple(col),markeredgecolor='k', markersize=6)
ax.set_title('DBSCAN Clustering Pulls Out Dense EV Charging Station Locales',size=20)
ax.tick_params(axis='x', size= 14)
ax.tick_params(axis='y', size= 14)

ax.set_xlabel('Longitude (deg)',size=16)
ax.set_ylabel('Longitude (deg)',size=16)


plt.show()


# In[50]:


#look at noise from clustering to see where the less dense EV charging stations are in relation to truck stops and turbines
#I currently have limited training data(not many truck stops) so I will target a linear regression on the most disperse current EV Stations
#which is the noise from the DBSCAN clustering.  
#However, once I collect more data on roads, traffic, truck stops, etc. I can filter out only the densest cluster or 
#or the top two densest clusters
fig,ax=plt.subplots(1,1,figsize=(10,10))

ax.scatter(texas_e_locs['xlong'],texas_e_locs['ylat'],c='#91bfdb',label='Dense EV Charging Stations')
ax.scatter(texas_turb_locs['xlong'],texas_turb_locs['ylat'], c='#fc8d59',label='Existing Wind Turbines')
ax.scatter(texas_truckstop_locs['xlong'],texas_truckstop_locs['ylat'],c='#d73027',label='Existing Truck Stops')
ax.scatter(xy.iloc[:, 0], xy.iloc[:, 1],color='k',label='Dispersed EV Charging Stations')
ax.legend(prop={'size': 14})
ax.tick_params(axis='x', size= 14)
ax.tick_params(axis='y', size= 14)

ax.set_xlabel('Longitude (deg)',size=16)
ax.set_ylabel('Longitude (deg)',size=16)

ax.set_title('Noise from DBSCAN Clustering Represents Dispersed EV Stations',size=20)
plt.show()


# In[51]:


texas_e_locs.value_counts('cluster')
#seeing how many EV stations are in each cluster and in the noise group (-1)


# In[52]:


target=X[class_member_mask & ~core_samples_mask]
#I currently have limited training data(not many truck stops) so I will target a linear regression on the most disperse current EV Stations
#which is the noise from the DBSCAN clustering.  
#However, once I collect more data on roads, traffic, truck stops, etc. I can filter out only the densest cluster or 
#or the top two densest clusters
print(len(target))
train_set=pd.DataFrame()
train_turb=texas_turb_locs.sample(n = len(target))
train_truck=texas_truckstop_locs.sample(n = len(target))
#randomize and downsample to match target set length
train_set['turbine_xlong']=list(train_turb['xlong'])
train_set['turbine_ylat']=list(train_turb['ylat'])
train_set['truckstop_xlong']=list(train_truck['xlong'])
train_set['truckstop_ylat']=list(train_truck['ylat'])


# In[53]:


features=['xlong','ylat']
from sklearn.linear_model import LinearRegression
reg = LinearRegression().fit(train_set, target)
predictions=reg.predict(train_set)


# In[54]:


xlong_predict=[val[0] for val in predictions] 
ylat_predict=[val[1] for val in predictions] 


# In[55]:


fig,[ax1,ax2]=plt.subplots(2,1,figsize=(10,20))
ax1.scatter(texas_e_locs['xlong'],texas_e_locs['ylat'],c='#91bfdb',label='Existing EV Charging Stations')
ax1.scatter(xlong_predict, ylat_predict,c='#4575b4',label='Predicted EV Charging Stations')
ax2.scatter(texas_turb_locs['xlong'],texas_turb_locs['ylat'], c='#fc8d59',label='Existing Wind Turbines')
ax2.scatter(texas_truckstop_locs['xlong'],texas_truckstop_locs['ylat'],c='#d73027',label='Existing Truck Stops')
ax2.scatter(xlong_predict, ylat_predict,c='#4575b4',label='Predicted EV Charging Stations')
ax1.legend(prop={'size': 14})
ax1.tick_params(axis='x', size= 14)
ax1.tick_params(axis='y', size= 14)
ax2.legend(prop={'size': 14})
ax2.tick_params(axis='x', size= 14)
ax2.tick_params(axis='y', size= 14)
ax1.set_xlabel('Longitude (deg)',size=16)
ax1.set_ylabel('Longitude (deg)',size=16)
ax2.set_xlabel('Longitude (deg)',size=16)
ax2.set_ylabel('Longitude (deg)',size=16)
ax1.set_title('Preliminary Lin. Reg. Places New EV Stations Outside of Existing Dense Clusters',size=20)
ax2.set_title('Preliminary Lin. Reg. Places New EV Stations Closer to Wind Turbines',size=20)
plt.show()


# In[56]:


#This approach needs more variables to train on, especially highway locations and traffic information
#However, this validates that combining methods such as density clustering and linear regression
#can place new EV charging outside of where there are already many stations in place and 
#closer to green energy resources such as existing wind turbines

