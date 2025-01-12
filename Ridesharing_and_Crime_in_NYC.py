# %%
---
title: "Ridesharing and Crime in New York City"
subtitle: Team name
author: Jack McNally, Eli Nacar, Samuel Sword, Tess Wanger
date: 12/05/2022
number-sections: true
abstract: _This report examines the relationship between reported crime density and ridesharing in New York City from 2014 to 2015. We first analyzed the popularity of Uber in each borough to determine where Uber is most used. Then, we looked at rideshare statistics over time to determine how the total number of ridesharing trips have changed. We also looked at the average use of rideshares over the course of a day to determine which times are more popular for ridesharing than others. After that, we examined reported crime locations in New York City to determine which areas were likely to be the safest for rideshare pick-ups and drop-offs. Finally, we looked at the correlation between Uber pick-up and drop-off locations and reported crime density to determine whether rideshare users, rideshare companies, and the NYPD should be concerned about the trend between crime and ridesharing. Although the correlation between the two was low, we recommended that rideshare users be more cognizant when waiting for rides in Manhattan, rideshare companies should advertise more night services when crime is typically higher, and the NYPD should focus its crime fighting efforts on areas in Manhattan where ridesharing is popular to make transportation safer_.
format: 
  html:
    toc: true
    toc-title: Contents
    code-fold: true
    self-contained: true
    font-size: 100%
    toc-depth: 4
    mainfont: serif
jupyter: python3
---

# %%
import pandas as pd
import json
import geopandas as gpd
import seaborn as sns
import plotly.express as px
import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk

# %% [markdown]
# ## Background / Motivation
# When thinking about potential topics for this project, we looked towards our personal experiences. As Northwestern students without cars, navigating Evanston and Chicago can be difficult. At times, the inflexibility of public transportation prevents students from accessing parts of the city and makes returning home after a night out more difficult. For this reason, rideshare apps have become increasingly popular among college students and for those living in big cities without cars. Since the founding of Uber in 2009, and Lyft in 2012, the use of rideshare apps has skyrocketed, allowing people to travel from point A to point B with the click of a button. Rideshare apps also offer a safety element, allowing users to avoid walking through unknown and potentially dangerous areas at the end of the night. When thinking about our own experiences as Northwestern students, walking back from parts of Chicago at the end of the night is not only unfeasible, but can be unsafe depending on where you are. For this reason, the specific use of rideshare apps to avoid walking through areas with higher crime, became the focus of our project as we sought to evaluate whether this relationship generalized to the larger population of people in NYC.

# %% [markdown]
# ## Problem statement 
# The four questions tackled in this project were selected in a way so that each question built on one another. The first two questions in our analysis were, “Where are Ubers used the most in New York City by borough?” and “How have the total number of Uber trips changed annually and changed in distribution among the hours of the day?”. The goal of addressing these two questions first was to get an understanding of how rideshare apps were used across all of New York City. A key part to the second question we tackled involved looking at how many Ubers were ordered on an hourly basis throughout the day. This allowed us to correlate the amount of rides and crime from a perspective other than the geographic approach based on boroughs. The hourly approach in the second question removed the influence of geography and based the relationship between ridesharing and crime purely on the time of day.
# 
# The third question we set out to answer was, “Which areas in NYC, by borough, have the most crime per hour?”. This question began our analysis of the publicly available crime data. With an understanding of rideshare trends established, crime data could be interpreted in the context of rideshare trends. Furthermore, by analyzing crime data as the number of crimes per hour by borough, connections could be made to the rideshare data from both the geographic and hourly perspectives.
# 
# The final question served as the main idea for our project. “What is the relationship between rideshare use and crime density by borough and hours of the day?”, pulled conclusions from all three of the analyses we had previously performed. The synthesis required for this final part of the project would measure the correlation between rideshare use and crime density from geographic and hourly perspectives. 

# %% [markdown]
# ## Data sources
# https://www.kaggle.com/datasets/fivethirtyeight/uber-pickups-in-new-york-city:
# This is an open access dataset with trip information for all Ubers, Lyfts, and various other rideshares in New York City throughout 2014 and 2015. For our analysis, we utilized information concerning the locations of various rideshare pickup locations.
# 
# https://data.cityofnewyork.us/Public-Safety/NYC-crime/qb7u-rbmr
# This is an open access dataset containing various information for police complaints filed in New York City during 2017. For our analysis, we utilized information concerning the time/date of crimes, type of crimes, and where crimes occurred.
# 
# https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page 
# This is a dataset made by the Taxi and Limousine Commission in New York City including unique location IDs and their corresponding taxi zones, service zones, and NYC borough.
# 
# https://data.cityofnewyork.us/Transportation/FHV-Base-Aggregate-Report/2v9c-2k7f 
# Open access dataset containing monthly reports of total dispatched trips, total dispatched shared trips, and unique dispatched vehicles aggregated by each FHV (For-Hire Vehicle) base. This dataset includes information on rideshare companies.

# %% [markdown]
# ## Stakeholders
# Rideshare Companies: If we find a correlation between pickup locations and crime density, rideshare companies will better understand where to focus their marketing campaigns as well as the most impactful areas to deploy rideshare drivers.
# 
# New York City: Locating the most dense crime locations is useful for the NYPD when deploying officers or simply better understanding the landscape of where crimes typically occur.
# 
# Rideshare Users: If a strong correlation is found between rideshare locations and crime density, rideshare users may be more inclined to bring their own transportation so as to not linger while waiting for a rideshare in an area with a higher crime density.

# %% [markdown]
# ## Data quality check / cleaning / preparation 

# %% [markdown]
# ##### TLC Aggregate Report Data

# %%
TLC_aggregate_report = pd.read_csv('FHV_Base_Aggregate_Report_20240926.csv')
print('Continuous variables:')
TLC_aggregate_report[['Total Dispatched Trips', 'Total Dispatched Shared Trips', 'Unique Dispatched Vehicles']].describe()

# %%
print('Categorical variables:')
TLC_aggregate_report[['Year', 'Month']].describe()

# %%
print('Missing values for each column')
TLC_aggregate_report.isnull().sum()

# %%
print('Number of unique values for year:', len(TLC_aggregate_report.Year.unique()))
print('Number of unique values for month:', len(TLC_aggregate_report.Month.unique()))

# %% [markdown]
# In the TLC aggregate report data, there are no abnormalities in any of the columns that require filtering, imputation, cleaning, etc. While the column 'DBA' does have about 42,000 missing values, this is acceptable. 'DBA' stands for 'Doing Business As'. When the values of 'Base Name' and 'DBA' are identical, the 'DBA' is left as NULL.
# No other data preparation was required for this dataset.

# %% [markdown]
# ##### Kaggle Dataset
# This dataset was originally split into multiple datasets. There were six separate sets for the 2014 data, and one for the 2015 data. Firstly, some data preparation had to be done in order to combine all of these datasets into one, which consisted of importing the data and concatenating all of the data sets into one dataframe:

# %%
#importing and concatonating all data from 2014 together
one = pd.read_csv('uber-raw-data-apr14.csv')
two = pd.read_csv('uber-raw-data-aug14.csv')
three = pd.read_csv('uber-raw-data-jul14.csv')
four = pd.read_csv('uber-raw-data-jun14.csv')
five = pd.read_csv('uber-raw-data-may14.csv')
six = pd.read_csv('uber-raw-data-sep14.csv')
frames = [one, two, three, four, five, six]
kaggle2014 = pd.concat(frames)
uber_data2014 = kaggle2014.copy()
kaggle2014 = kaggle2014.reset_index()

#adding 2015 data
kaggle2015 = pd.read_csv('uber-raw-data-janjune-15.csv')
kaggle_2014_to_2015 = pd.concat([kaggle2014, kaggle2015], axis=0)
kaggle_2014_to_2015.reset_index()

# %% [markdown]
# After concatenating the datasets, the columns which contained the pickup times for each data group (2014 and 2015) were separated. These two columns were combined into one Series, and then added to the dataframe as 'Pickup Time'. Finally, 'Pickup Time' was converted to the datetime datatype, which would be necessary for the analysis in the future. A similar process was done for 'Dispatch Base', except that for this column, it was left as a string.

# %%
#cleaning pickup time columns
pickup_time_2014 = kaggle2014['Date/Time']
pickup_time_2015 = kaggle2015['Pickup_date']

pickup_times = pd.concat([pickup_time_2014, pickup_time_2015], axis=0)
pickup_times = pickup_times.reset_index()
pickup_times = pickup_times.drop('index', axis = 1)
pickup_times = pickup_times.rename(columns={0: "Pickup Time"})

#adding this combined pickup time to kaggle2014-2015, then converting to datetime
kaggle_2014_to_2015['Complete Pickup Time'] = pickup_times
kaggle_2014_to_2015['Complete Pickup Time'] = pd.to_datetime(kaggle_2014_to_2015['Complete Pickup Time'], format='mixed')

#cleaning base columns
base_2014 = kaggle2014['Base']
base_2015 = kaggle2015['Dispatching_base_num']

bases = pd.concat([base_2014, base_2015], axis=0)
bases = bases.reset_index()
bases = bases.drop('index', axis = 1)
bases = bases.rename(columns={0: "Base"})

#adding combined dispatch to kaggle2014-2015
kaggle_2014_to_2015['Dispatch Base'] = bases

# %% [markdown]
# Lastly, the 'Index', 'Date/Time', 'Base', 'Dispatching_base_num', 'Pickup_date', and 'Affiliated_base_num' columns were dropped from the DataFrame.  In the final dataset, the first portion of the observations correspond to the 2014 data, while the second portion corresponds to the 2015 data.

# %%
columns_drop=['index', 'Date/Time', 'Base', 'Dispatching_base_num', 'Pickup_date', 'Affiliated_base_num']
kaggle_2014_to_2015 = kaggle_2014_to_2015.drop(columns=columns_drop)
kaggle_2014_to_2015

# %%
kaggle_2014_to_2015.describe()

# %%
print('Missing values for each column')
kaggle_2014_to_2015.isnull().sum()

# %%
print('Number of unique values for Dispatch Base:', len(kaggle_2014_to_2015['Dispatch Base'].unique()))

# %% [markdown]
# The missing values in this DataFrame are due to the fact that the observations pertaining to the 2014 data do not have values for locationID, while the observations pertaining to the 2015 data do not have latitude and longitude data.

# %% [markdown]
# ##### NYPD Complaint Data Historic

# %%
crime_data = pd.read_csv('NYPD_complaint_data.csv')
#dropping unneccesary columns
crime_data = crime_data.drop(columns=['Unnamed: 0', 'complaint_year', 'complaint_month'])

# %% [markdown]
# To clean the data, I started by renaming the columns so that it was easier for me to identify what information each column contained.

# %%
new_col_names = ['complaint_date', 'complaint_time', 'gen_description', 'pd_description', 'level_of_offense', 'borough', 'location_type', 'Latitude', 'Longitude']
crime_data.columns = new_col_names

# %% [markdown]
# After that, I converted the date of the reported crime to datetime so that I could sort out the reported crimes from 2014 and 2015. I also used this to get the month and year of each reported crime so that I would be able to sift through the data using the conditions month and year.

# %%
crime_data['complaint_date'] = pd.to_datetime(crime_data['complaint_date'], errors='coerce')
crime_data['complaint_year'] = crime_data['complaint_date'].dt.strftime('%Y')
crime_data['complaint_month'] = crime_data['complaint_date'].dt.strftime('%m')

# %% [markdown]
# I also narrowed it down to only include data from 2014 and 2015 - the time period our group decided to analyze.

# %%
data_14_15 = crime_data[(crime_data['complaint_year']=='2014')|(crime_data['complaint_year']=='2015')]

# %% [markdown]
# I needed the longitude and latitude because I was plotting points on a map based on those coordinates. I decided to drop all of the rows that had no data for longitude or latitude. Before making this choice, I was concerned about affecting the conclusion because there were 26 rows with missing data. However, the data with missing coordinate values only made up about 0.002% of the total data which has 971,051 observations. I also did not want to impute any coordinate data because there were still plenty of data points to demonstrate the trend, and I could not determine a way to impute the data that would make sense.

# %%
data_14_15.shape

# %%
print('Number of missing values by column:')
data_14_15.isna().sum()

# %%
data_14_15 = data_14_15[~data_14_15.Longitude.isna()]
data_14_15 = data_14_15[~data_14_15.Latitude.isna()]

# %%
crime_14_15 = data_14_15.copy()

# %% [markdown]
# While there are still missing values, they do not impact the analysis.

# %%
print('Number of missing values by column:')
data_14_15.isna().sum()

# %%
print('Number of unique values by column:')
data_14_15.apply(lambda x: len(x.unique()))

# %%
data_14_15.describe()

# %% [markdown]
# ##### Borough Boundaries Geojson

# %%
boros = gpd.read_file('Borough_Boundaries.geojson')

# %% [markdown]
# This dataset was very easy to work with because there were only 5 total entries: one for each of the boroughs in New York City. The describe feature worked slightly differently on this dataset because it is a Geodataframe. I did not have to alter this dataset because it was very straightforward and contained all the data I needed.

# %%
boros.describe()

# %%
print('Number of missing values by column:')
boros.isna().sum()

# %% [markdown]
# ### Analysis 1
# *By Jack McNally*

# %%
uber_apr=one.copy()
uber_aug=two.copy()
uber_jul=three.copy()
uber_jun=four.copy()
uber_may=five.copy()
uber_sep=six.copy()

# %% [markdown]
# For my analysis, I analyzed the number of Ubers ordered in New York City broken down into each of the five boroughs (Manhattan, Bronx, Queens, Staten Island, Brooklyn). However, because the Uber data was structured differently for 2014 and 2015, my approach in categorizing Uber rides to boroughs was different for each dataset.
# 
# In the 2014 Uber data, each ride included information on the longitude and latitude of the pickup location. With this information, the biggest challenge I initially faced was figuring out how to use the geographic coordinates of a pickup location to then identify which NYC borough it was located in. At first, I researched whether there was any publicly available Python code that could accept longitude and latitudes and return the NYC borough. After looking extensively on Stack Overflow and GitHub, the only code that potentially seemed helpful was a function that accepted longitude and latitudes and returned the NYC neighborhood for that location. I reasoned that if I could find a data set matching each neighborhood to a borough, then I could combine the function and this data set to assign locations to their respective boroughs. As an alternative approach, I also considered mapping out each borough using rectangles defined by ranges of longitude and latitudes. If a set of coordinates fell within the rectangle, then that location would be assigned to that borough. Looking at these two approaches side by side, I decided to map the boroughs as it seemed like a more basic approach that would result in less errors. Additionally, I was unable to find a dataset linking NYC neighborhoods to boroughs. 
# 
# For the mapping approach, I took screenshots of each borough and planned out how to efficiently cover the area of a borough while minimizing overlap with other boroughs.

# %% [markdown]
# After mapping out each borough, I used Google maps to identify the longitude and latitudes of the sides of the rectangles I used to cover the area of the boroughs. Using these rectangle coordinates, I defined multiple ranges for each borough that would return whether or not a specific set of coordinates fell within the preselected ranges.

# %%
# extension = 'csv'
# all_filenames = [i for i in glob.glob('*.{}'.format(extension))]
# combined_uber=pd.concat([pd.read_csv(f) for f in all_filenames])
# combined_uber.to_csv("combined_csv.csv", index=False, encoding = 'utf-8-sig')
combined_uber = uber_data2014.copy()
combined_uber

# %%
%matplotlib inline
data = pd.DataFrame()

#(down, up) (left, right)

def manhattan(data):
    data = data[(data['Lat'].between(40.701426, 40.810606) & data['Lon'].between(-74.019561, -73.975601)) |
                #square one
                (data['Lat'].between(40.775555, 40.878306) & data['Lon'].between(-73.975601, -73.930901)) |
                #square 2
                (data['Lat'].between(40.756607, 40.775555) & data['Lon'].between(-73.975601, -73.945157)) |
                #square 3
                (data['Lat'].between(40.734383, 40.756607) & data['Lon'].between(-73.975601, -73.962062))
                #square 4 
               ]
    return data

def bronx(data):
    data = data[(data['Lat'].between(40.802951, 40.912623) & data['Lon'].between(-73.910130,-73.764666)) |
                #square one
                (data['Lat'].between(40.802951, 40.859068) & data['Lon'].between(-73.930324,-73.910130))
                #square two 
            ]
    return data

def queens(data):
    data = data[(data['Lat'].between(40.621930,40.800482) & data['Lon'].between(-73.868944,-73.700464))|
                #square one
                (data['Lat'].between(40.683894,40.790877) & data['Lon'].between(-73.913135,-73.868944))|
                #square two
                (data['Lat'].between(40.729766,40.781454) & data['Lon'].between(-73.934580,-73.913135))
                #square three
               ]
    return data

def brooklyn(data):
    data = data[(data['Lat'].between(40.572077,40.695474) & data['Lon'].between(-74.042047,-73.858306))|
                #square one
                (data['Lat'].between(40.695474,40.708849) & data['Lon'].between(-74.000258,-73.912798))|
                #square two
                (data['Lat'].between(40.708849,40.734609) & data['Lon'].between(-73.970169,-73.922498))
                #square three
               ]
    return data

def staten_island(data):
    data = data[(data['Lat'].between(40.494395,40.648781) & data['Lon'].between(-74.205546,-74.052424))|
                #square one
                (data['Lat'].between(40.494395,40.571633) & data['Lon'].between(-74.256701,-74.205546))
                #square two
                ]
    return data
stat = staten_island(combined_uber)
bron = bronx(combined_uber)
manh = manhattan(combined_uber)
quee = queens(combined_uber)
brook = brooklyn(combined_uber)

# %%
stat

# %%
bron

# %%
manh

# %%
quee

# %%
brook

# %%
borough_data = pd.DataFrame()
borough_data['Borough'] = "Staten Island", "Bronx", "Manhattan", "Queens", "Brooklyn"
borough_data['Uber Rides'] = len(stat), len(bron), len(manh), len(quee), len(brook)
borough_data['Uber Rides'].sum()

# %% [markdown]
# The uber data was broken down into months of the year. After combining the months together, I used the borough-defining function, identified which pickup locations belonged to each borough, and then added the number of rides for each borough to make a comprehensive comparison for the 2014 Uber data.

# %% [markdown]
# For the 2015 Uber data, each ride had a specific taxi zone location number. Initially I was unsure how I could link the taxi zone location number to NYC borough. However, after some research, I found that the taxi zone location numbers were standardized for rideshare apps and taxis across the city. I found a dataset linking each location number to other information including NYC borough, and then merged the 2015 Uber data and the taxi zone location data and added the number of rides for each borough [1].

# %%
uber_jan_jun_15 = kaggle2015.copy()
uber_jan_jun_15.sort_values('locationID').head()

# %%
taxi_zone= pd.read_csv('taxi_zone_lookup.csv')
taxi_zone.sort_values('LocationID').head()

# %%
uber_jan_jun_15 = pd.merge(taxi_zone, uber_jan_jun_15, left_on= 'LocationID', right_on= 'locationID')
uber_jan_jun_15.head()

# %%
del uber_jan_jun_15['locationID']
#del uber_jan_jun_15['service_zone']

# %%
uber_jan_jun_15['Borough'].value_counts()

# %%
manhattan_15= uber_jan_jun_15[uber_jan_jun_15['Borough']=='Manhattan']
brooklyn_15= uber_jan_jun_15[uber_jan_jun_15['Borough']=='Brooklyn']
queens_15= uber_jan_jun_15[uber_jan_jun_15['Borough']=='Queens']
bronx_15= uber_jan_jun_15[uber_jan_jun_15['Borough']=='Bronx']
staten_island_15= uber_jan_jun_15[uber_jan_jun_15['Borough']=='Staten Island']

# %%
borough_data_15= pd.DataFrame()
borough_data_15['Borough'] = "Staten Island", "Bronx", "Manhattan", "Queens", "Brooklyn"
borough_data_15['Uber Rides'] = len(staten_island_15), len(bronx_15), len(manhattan_15), len(queens_15), len(brooklyn_15)

# %%
borough_data_15

# %% [markdown]
# To visualize the distribution of the total number of rides across boroughs between 2014 and 2015, I used a bar plot because NYC boroughs are categorical variables.

# %%
ax1 = sns.barplot(x= 'Borough', y = 'Uber Rides', data = borough_data_15)
ax1.figure.set_figwidth(10)
plt.xlabel('Borough', fontsize = 14)
plt.ylabel('Number of Uber Rides', fontsize = 14)

# %%
borough_data_15['Uber Rides'].sum()

# %% [markdown]
# The biggest problem I anticipated was with my methodology for the 2014 Uber data. As mentioned before, I was unsure how I could attribute a location in NYC to its borough. In looking at what information was available, I realized that using rectangles to cover the area of each borough was the best approach. However, with this method I found that there would be gaps and misclassification of Uber pickups in certain locations due to the rigidity of only using rectangles to represent each borough. To address this, I made a deliberate step in understanding how to cover exclusively the area of each borough while using multiple rectangles. After creating the code to identify a borough, I would test the method using locations that could present difficulties for my approach. Although the first rectangle method I used was ultimately successful, this came as a result of careful planning and anticipation of the problems I would face with my analysis.
# 
# As mentioned earlier, the only code I used in my analysis outside of the original Uber datasets was the taxi zone dataset I used for the 2015 Uber ride information [1]. The changes I made to this code involved merging it to the 2015 data and removing all columns besides the borough classification.

# %% [markdown]
# ### Analysis 2
# *By Samuel Sword*

# %% [markdown]
# For this part of the project, change in rideshare statistics over time was analyzed, using the Kaggle dataset consisting of rideshare statistics in New York City and the TLC aggregate data consisting of Vehicle For Hire (VFH) statistics. Ridesharing companies are included in this dataset. Firstly, the TLC aggregate data was analyzed in order to compare and contrast the rideshare industry with the traditional transportation industry (taxis, limos, etc). A boxplot based on the entire dataset was created in order to visualize the average number of dispatches per month, per year. It is worth noting that in this dataset, each observation corresponds to an entire month's worth of data for the company which the observation represents.

# %%
TLC_aggregate_report = pd.read_csv('FHV_Base_Aggregate_Report.csv')

# dispatched trips by year
ax = sns.barplot(x="Year", y = 'Total Dispatched Trips',  data=TLC_aggregate_report)
ax.figure.set_figwidth(15)
plt.xlabel('Year', fontsize=14);
plt.ylabel('Avg dispatched trips per month', fontsize=14)
ax.set_title("Average Dispaches Per Year", fontsize = 20)

# %% [markdown]
# As can be seen in the graph above, the average number of dispatches per month, per year has consistently increased, with the exception of the years between 2019 and 2020. It is almost certain that the impact COVID-19 is the primary cause of this plummet in dispatches, and it can be assumed that the trend of positive growth would have continued if not for the effect of the pandemic, as it showed no signs of slowing down beforehand, and continued its growth the following year. This bar graph gives valuable insight into the general landscape of VFH dispatches, as it indicates that the industry has seen positive growth for the last 7 years. 
# 
# In order to approximate how rideshare companies have affected this positive growth, the four most popular rideshare companies (Uber, LYFT, Juno, and Via) were removed from the dataset. The following barplot visualizes the average non-rideshare dispatches per month, per year:

# %%
#removing top four rideshare
TLC_no_rideshare = TLC_aggregate_report.loc[(TLC_aggregate_report['Base Name'] != 'UBER') &
                                            (TLC_aggregate_report['Base Name'] != 'LYFT')&
                                            (TLC_aggregate_report['Base Name'] != 'JUNO') &
                                            (TLC_aggregate_report['Base Name'] != 'VIA')]

ax = sns.barplot(x="Year", y = 'Total Dispatched Trips',  data=TLC_no_rideshare)
ax.figure.set_figwidth(8)
plt.xlabel('Year', fontsize=14);
plt.ylabel('Avg dispatched trips (per month)', fontsize=14)
ax.set_title("Average non-Rideshare Dispaches Per Year", fontsize = 20)

# %% [markdown]
# A stark contrast can be found between this barplot and the first barplot: whereas the former bar plot reveals an overall positive trend, the latter bar plot reveals an overall negative trend. In other words, the two bar plots above suggest that a primary factor in the growth of average VFH dispatches is the rideshare industry. To reinforce this claim, the top four rideshare companies were filtered into their own dataset, and a bar plot of average dispatches per month per year of these companies was created:

# %%
#UBER LIFT JUNO AND VIA
UBER_LIFT_JUNO_VIA = TLC_aggregate_report.loc[(TLC_aggregate_report['Base Name']=='UBER') | 
                                              (TLC_aggregate_report['Base Name']=='LYFT') |
                                              (TLC_aggregate_report['Base Name']=='JUNO') |
                                              (TLC_aggregate_report['Base Name']=='VIA')]

# Dispatched trips by year
ax = sns.barplot(x="Year", y = 'Total Dispatched Trips',  data=UBER_LIFT_JUNO_VIA)
ax.figure.set_figwidth(15)
plt.xlabel('Year', fontsize=14);
plt.ylabel('Avg dispatched trips per month', fontsize=14)
ax.set_title("Top 4 Rideshare Dispaches Per Year", fontsize = 20)

# %% [markdown]
# This bar plot shows a positive trend in rideshare dispatches over time for the top four rideshare companies (aside from 2019 to 2020), which supports the statement that rideshare companies have primarily contributed to the positive growth of VFH companies. Furthermore, if we examine the range of values that both bar plots have in their y-axis, it is discovered that the number of non-rideshare VFH dispatches, whose barplot has a range on the y axis of 0 to 7000 is much lower than the number of rideshare VFH dispatches, whose barplot has a range on the y-axis of 0 to 10,000,000. This shows that rideshare companies are operating with a much larger client base than traditional VFH companie, as can be seen by the line plot below:

# %%
#getting all the VFH companies that aren't top four rideshare
VFH_comparison = TLC_aggregate_report.copy()
rideshare_companies_list = ['UBER', 'LYFT', 'JUNO', 'VIA']
VFH_comparison = VFH_comparison.assign(Rideshare_Company = lambda x: x['Base Name'].isin(rideshare_companies_list))

# pivoting tables
rideshare_norideshare_comparison = VFH_comparison.pivot_table(index = 'Year', columns = 'Rideshare_Company',values = 'Total Dispatched Trips')

ax = rideshare_norideshare_comparison.plot(ylabel = 'Total Dispatched Trips',figsize = (10,6),marker='o')
ax.yaxis.set_major_formatter('{x:,.0f}')
plt.xlabel('Year', fontsize=14);
plt.ylabel('AVG dispatched trips per month', fontsize=14)
ax.set_title("Rideshare vs. Non-Rideshare Companies", fontsize = 20)

# %% [markdown]
# Note that in the graph above, the value of the line plot, which represents non-rideshare companies, is not equal to 0 for each year. Instead, compared to the large number of cars that rideshare companies are dispatching, the number of cars that non-rideshare companies are dispatching is a very small number.
# 
# More specifically, if we examine Uber and LYFT dispatches on their own, we see an even higher range of 0 to 12,000,000:

# %%
# Just UBER and LIFT data
TLC_agg_UBER_and_LIFT = TLC_aggregate_report.loc[(TLC_aggregate_report['Base Name']=='UBER') | (TLC_aggregate_report['Base Name']=='LYFT')]

# dispatched trips by year
ax = sns.barplot(x="Year", y = 'Total Dispatched Trips',  data=TLC_agg_UBER_and_LIFT)
ax.figure.set_figwidth(8)
plt.xlabel('Year', fontsize=14);
plt.ylabel('Avg dispatched trips (per month)', fontsize=14)
ax.set_title("Average Uber/LYFT Dispaches Per Year", fontsize = 20)

# %% [markdown]
# Finally, the dataset consisting of the top four rideshare companies was reshaped by pivoting the table, which allowed for the visualization of each of the four’s dispatch trends over time:

# %%
top4_by_year = UBER_LIFT_JUNO_VIA.pivot_table(index = 'Year', columns = 'Base Name',values = 'Total Dispatched Trips')
#lineplot
ax = top4_by_year.plot(ylabel = 'Total Dispatched Trips',figsize = (10,6),marker='o')
ax.yaxis.set_major_formatter('{x:,.0f}')
ax.set_title("Top Rideshare Companies by Year", fontsize = 20)

# %% [markdown]
# Here, it can be seen that Uber is the dominant rideshare company in New York City by a large margin. For example, in 2019, Uber averaged 14,000,000 dispatches per month, whereas the next most popular rideshare company, LYFT, averaged about 5,000,000. Also, through this line plot, it is revealed that Juno and Via don’t have available data for all of the years that the data covers.
# 
# Following the analysis of the TLC aggregated dataset, an examination of the kaggle dataset consisting of Uber rides from 2014-2015 in NYC was conducted. This dataset included specific times of pickups for each dispatch, so the hope was to analyze rideshare statistics by time of day/night. The data from 2014 was broken into monthly sets, so first, those datasets were imported and concatenated together, then concatenated with the 2015 data to create a dataset with the entire time frame. The columns consisting of pickup times were combined and transformed into a datetime datatype, and from here a bar plot reflecting number of Ubers dispatched by hour was created:

# %%
#plotting average pickup time
kaggle_2014_to_2015['Complete Pickup Time'].dt.hour.value_counts().sort_index().plot(kind = 'bar', rot=0, color='orange', figsize=(12,4))
plt.xlabel('Hour', fontsize=14)
plt.ylabel('Number of Ubers called', fontsize=14)
plt.title('Uber Dispatches by Hour', fontsize=18)

# %% [markdown]
# Based on the visualization above, Uber dispatches in New York have two relative peaks: one at 8 AM, and one at 6 PM. An explanation for the first peak could be that it corresponds to the times when individuals are ordering Ubers in order to get to work, while the second is likely the time when individuals are ordering Ubers in order to get home from work. Furthermore, it makes sense that the second peak at 6 PM is significantly higher than the peak at 8 AM, as people are probably also ordering Ubers in order to get to various evening events.
# 
# Below, a plot of average Ubers dispatched daily in a month is visualized. There aren’t any trends in the graph that suggest any significant meaning, but it is worth noting that the 31 days of the month sees much lower Uber dispatches than the rest of the days. This could likely be due to the fact that not all months have 31 days, which might affect how the graph visualizes the average for that day.

# %%
#plotting average pickups per day
kaggle_2014_to_2015['Complete Pickup Time'].dt.day.value_counts().sort_index().plot(kind = 'bar', rot=0, color='orange')
plt.xlabel('Day', fontsize=14)
plt.ylabel('Ubers called', fontsize=14)

# %% [markdown]
# Lastly, a pre-cleaned dataset consisting of crime complaint in New York City with the same timeline as the Kaggle dataset was imported, and a barplot visualizing crime complaints by hour was created:

# %%
crime_14_15['complaint_time'] = pd.to_datetime(crime_14_15['complaint_time'])

#crime complaints by hour
crime_14_15['complaint_time'].dt.hour.value_counts().sort_index().plot(kind = 'bar', rot=0, figsize=(15,5))
plt.xlabel('Hour', fontsize=14)
plt.ylabel('Complaints', fontsize=14)
plt.title('Crime Complaints by Hour', fontsize=18)

# %% [markdown]
# Here is the bar plot of Uber dispatches per hour for comparison:

# %%
kaggle_2014_to_2015['Complete Pickup Time'].dt.hour.value_counts().sort_index().plot(kind = 'bar', rot=0, color='orange', figsize=(12,4))
plt.xlabel('Hour', fontsize=14)
plt.ylabel('Number of Ubers called', fontsize=14)
plt.title('Uber Dispatches by Hour', fontsize=18)

# %% [markdown]
# Unsurprisingly, the two graphs have very similar shapes, and have peaks around 6 PM. This is to be expected, not because crime rates and and Uber dispatches are causally correlated, but because the frequency of both crime complaints and Uber dispatches are most likely correlated to times when the most human activity in New York City is occurring in general (this would likely be around evening time). Furthermore, as general human activity decreases into the late night/early morning hours, so do crime complaints and Uber dispatches.
# 
# In doing this analysis, an anticipated issue was that of timeline differences, as the TLC data ranged from 2015-2022, while the Kaggle data only consisted of data on the years of 2014 and 2015. While the TLC dataset could have been pared down to a smaller time frame, it was decided that in this instance, time range differences were okay. Since Uber has seen steady and predictable growth in the years of the TLC dataset (aside from 2020), it can be assumed in 2014 (one of the years that the Kaggle dataset includes) Uber saw a growth in dispatches as well.
# 
# A problem that was encountered upon doing the analysis was that the TLC dataset did not have any indicator of which of the companies included in the dataset were ridesharing companies and which were traditional VFH companies. In order to solve this issue, external research was done to determine the four most popular rideshare companies in New York City [2]. Those four companies are Uber, LYFT, Juno, and Via. While this still leaves the issue that other rideshare companies might’ve been included in the data when filtering out those four companies, it was deemed okay to do so. The rest of the ridesharing companies would have a minimal effect on the data even if they were included in the filtered dataset consisting of rideshare data, as they would be significantly smaller in size compared to the other 4 companies (especially Uber and LYFT). One final issue that arose was that the data utilized in Kaggle dataset only consists of Uber statistics, while the TLC aggregated data consists of all documented rideshare companies. In this case, though, because Uber makes such a large portion of the rideshare landscape, the trends found in that dataset can be generalized to ridesharing as a whole.

# %% [markdown]
# ### Analysis 3
# *By Tess Wagner*

# %% [markdown]
# I determined which boroughs had the highest reported crime densities in New York City, and I looked for changes in crime densities throughout 2014 and 2015. To clarify, "crime density is most simply described as the number of crime incidents within a certain area" [3]. I began by cleaning my data. The dataset I used was titled "NYPD Complaint Data Historic" which I downloaded from the City of New York data site. This dataset included over seven million entries and over 30 columns, though I only imported 9 of the columns to make it run more efficiently.
# 
# I wanted to create density heatmaps to see which boroughs had the highest reported crime densities and if there were any changes in the data over the period 2014 to 2015. I split the data into four parts: January through June of 2014, July through December of 2014, January through June of 2015, and July through December of 2015. I used a lambda function to turn the data in the complaint month column from datetime objects into integers so that I could sort the data by month more easily.

# %%
data_14_15['complaint_month'] = data_14_15['complaint_month'].apply(lambda x:int(x))

# %%
data_2014_jantojune = data_14_15[(data_14_15['complaint_year']=='2014')&(data_14_15['complaint_month']<=6)]
data_2014_julytodec = data_14_15[(data_14_15['complaint_year']=='2014')&(data_14_15['complaint_month']>=7)]
data_2015_jantojune = data_14_15[(data_14_15['complaint_year']=='2015')&(data_14_15['complaint_month']<=6)]
data_2015_julytodec = data_14_15[(data_14_15['complaint_year']=='2015')&(data_14_15['complaint_month']>=7)]

# %% [markdown]
# At this stage, I ran into a problem when creating the maps. There were so many data points across the maps that the Jupyter notebook crashed and returned a memory error each time I tried to run it. To fix this problem, I first tried to break the data up into smaller groups. This did not work because all of the maps were still trying to parse through a massive amount of data in the same notebook. To circumvent this issue, I saved the 4 groups of data to csv files so that I could create each of the maps in different Jupyter Notebooks. Then, I created a new Jupyter notebook for each of the 4 groups of data so that I could create a density map for each period of time.
# 
# *(NOTE: for the purpose of this report, I did not create separate csv files and Jupyter notebook files. All of the code for the maps is in a cell with "#|eval:false" to prevent it from running and crashing the notebook. I have attached a zip file on Canvas with the html files including the code and the interactive map images so that you can easily access them.)*
# 
# To create these maps, I used a python graphing library called Plotly. I was looking for a way to efficiently plot using longitude and latitude, and I came across a Plotly article for making density heatmaps [4]. Using the longitudes and latitudes for each point in my dataset and the "density_mapbox" feature of Plotly, I created an interactive map with a zoom feature which allowed me to look closely at the different boroughs of New York City. My code was not too different from the sample code in the article because the purpose of my code and the code in the article were both to create a density heatmap using longitude and latitude. After looking at the Plotly documentation, I replaced the pieces of code with the data I needed, and I removed the parameter "z" because I did not need to weight any of my points. Overall, Plotly was a very handy tool for analyzing the data.
# 
# After that, I ran into some difficulty trying to plot outlines of the boroughs using Plotly. I came across the GeoPandas library which uses geojson files containing longitude and latitude to create map-like plots in matplotlib [5]. I created a GeoDataframe that contained Multipolygon objects that GeoPandas uses to create borders based on coordinates. Next I had to figure out how to transpose the borough borders onto the Ploty density heatmap. After trying many different ways to plot the points, I came across code in a Stack Overflow forum that used the parameter "mapbox" within the "update_layout" attribute of the figure [6]. After getting rid of the parameters that I did not need and adjusting the other to fit my data, I was able to transpose the data from the Geopandas object onto the maps to create the borough limits.

# %%
boros = gpd.read_file('Borough_Boundaries.geojson')

# %%
#|eval:False
list_of_data=[data_2014_jantojune,data_2014_julytodec,data_2015_jantojune,data_2015_julytodec]
for data in list_of_data:
    fig = px.density_mapbox(data, 
                            lat='Latitude', 
                            lon='Longitude', 
                            radius=1,
                            zoom=9,
                            mapbox_style="stamen-terrain")
    fig.update_layout(
        mapbox={
            "layers":[{
                "source":json.loads(boros.geometry.to_json()),
                "type": "line",
                "color": "black",
                "line": {"width": 2},
            }]
        }, 
        margin={"l": 0, "r": 0, "t": 0, "b": 0}
    )
    fig.show()

# %% [markdown]
# After identifying the edges of the boroughs, I compared changes in reported crime density across New York using the maps. I determined that Manhattan had the highest reported crime density across all four periods, and Staten Island had the lowest crime density across all four periods. There is not too much change across this time period which makes sense because the time period is not very long. However, the heat density maps made it difficult to compare the actual number of crimes in each borough. I created a column with just the year and month of each reported crime and converted the type to datetime so that I could create a countplot of the reported crimes separated by borough. This countplot showed that Brooklyn had the highest number of reported crimes across 2014 and 2015, and Staten Island had the lowest number of reported crimes by a large margin across 2014 and 2015.

# %%
data_14_15['year_month'] = data_14_15['complaint_date'].dt.strftime('%Y, %m')
data_14_15['year_month'] = pd.to_datetime(data_14_15['year_month'])

# %%
ax = sns.countplot(x=data_14_15.year_month.sort_values(), hue='borough', data=data_14_15)
ax.set_ylabel('Number of Reported Crimes', fontsize=20)
ax.set_xlabel('Crime by Month', fontsize=20)
ax.set_title('Number of Reported Crimes by Borough', fontsize=30)
ax.figure.set_figwidth(25)
ax.figure.set_figheight(8)
ax.set_xticklabels(['01/14','02/14','03/14','04/14','05/14','06/14','07/14','08/14','09/14','10/14','11/14','12/14',
                    '01/15','02/15','03/15','04/15','05/15','06/15','07/15','08/15','09/15','10/15','11/15','12/15'], fontsize=15)
ax.tick_params(axis='y', labelsize=15)

# %% [markdown]
# For this project, it is useful to look at reported crime density by borough to help determine the relationship between rideshare pick-up and drop-offs because rideshare users may feel safer in areas with lower reported crime densities and rideshare companies could use this information to find pick-up and drop-off areas where the reported crime density is lower. Areas with higher reported crime densities have higher concentrations of crime in that area. Comparing the visualizations, we can see that Brooklyn has about 2000 more reported crimes than Manhattan across all four periods. However, Manhattan has only one-third of the area of Brooklyn and is the smallest area of all the boroughs [7]. From this information, it makes sense that Manhattan has the highest reported crime density because it has the second highest amount of reported crimes and the smallest land area. Staten Island has the least amount of crimes by a large margin as well as the lowest reported crime density. One factor to consider is population density. Manhattan has the highest population density and Staten Island has the lowest population density [7]. While there cannot be a definitive conclusion made about the effect of population density on crime density, it makes sense that a higher population in a small land area would see a higher concentration of crimes in that land area. However, this does not mean that there is a greater proportion of reported crimes per person living there. This is only taking into account the area of the space in which the reported crimes are taking place. For this reason, areas in New York with lower population densities may have lower reported crime rates. Furthermore, if we are defining safety by reported crime density, Staten Island is the safest.
# 
# This analysis is important to answering the question "Is there a relationship between crime rates and ridesharing locations in NYC?" because it is useful to determine the areas with the least crime. Rideshare users would be interested to know which areas have lower crime densities because it may help them feel safer when using ridesharing services. Also rideshare The NYPD could also use this information for more effective crime fighting efforts. If they are focusing crime fighting efforts on areas where ridesharing pick-up and drop-offs are high to make transportation safer, it would be useful to know the reported crime density in that area. There are likely different plans of action for areas with high reported crime densities compared to low reported crime densities. Also, it may be easier to start in areas where reported crime is low and rideshare activity is high. This data can be cross referenced with ride sharing data to determine the relationship between the two and create a better experience for users and drivers.

# %% [markdown]
# ### Analysis 4
# *By Eli Nacar*
# 
# For this section of the analysis, we chose to look at whether there was a difference in correlation between uber pickup locations, Lyft pickup locations, and crime density in New York City. Further expanding the analysis to check correlation values between different rideshare pickup locations and crime density. If it was found that one type of rideshare was more highly correlated to crime than another, exploring the reason for this difference in a separate analysis would gleam useful information for city policy makers, rideshare companies, and even common riders if it was discovered why one rideshare may correlate to a higher density of crime than the other. To explore this question, this analysis looked at the locations of each individual crime, uber, and lyft pickup within New York City, ultimately drawing a comparison between their correlation values. We believed that if we were able to illustrate how many crimes, ubers, and lyfts occured in each latitude/longitude pair, we could draw relevant conclusions concerning the correlation between these three values as well as expand the analysis to each borough.

# %%
crime_data = crime_14_15.copy()
crime_data.rename(columns={'Latitude':'Lat', 'Longitude':'Lon'}, inplace=True)
crime_data.head()

# %%
### Importing LYFT Data
lyft_data = pd.read_csv('other-LYFT_B02510.csv')
lyft_data.drop('Unnamed: 3', axis=1, inplace=True)
lyft_data.rename(columns={'start_lat':'Lat', 'start_lng':'Lon'}, inplace=True)
lyft_data.head()

# %%
### UBER Data
uber_apr14 = one.copy()
uber_may14 = five.copy()
uber_jun14 = four.copy()
uber_jul14 = three.copy()
uber_aug14 = two.copy()
uber_sep14 = six.copy()

uber_datasets = [uber_apr14, uber_may14, uber_jun14, uber_jul14, uber_aug14, uber_sep14]
uber_data = pd.concat(uber_datasets)
uber_data.head()

# %% [markdown]
# The latitude and longitude values for each individual dataframe were not normalized to a certain level of precision. In order to standardize, we set each latitude and longitude value to three decimal points of precision (equal to about a 110 square meter plot of land). What this means, is that essentially every crime, uber, and lyft pickup spot is generalized to have occurred within a tennis court sized plot of land in New York City. We were comfortable with this level of precision as it allowed for meaningful analysis while not sacrificing relevant precision. 

# %%
### Standardize Lat and Long to 3 Decimal Points
# Three decimal point is worth 110 meters. 

def round_3(x):
    x['Lat'] = x['Lat'].apply(lambda x: round(x, 3))
    x['Lon'] = x['Lon'].apply(lambda x: round(x, 3))
    
round_3(crime_data)
round_3(lyft_data)
round_3(uber_data)

crime_data.head()

# %% [markdown]
# Once the data was cleaned and standardized, we had three data frames with information concerning the locations of every reported crime, uber, and lyft throughout 2014 and 2015 in New York City. Now, with this information, we could begin our actual analysis of the correlation between each value. 
# 
# To make our analysis simpler, I concatenated all of the individual data frames into a single dataframe called "locations." Locations have every single unique latitude and longitude value within the dataset with other columns indicating the number of crimes, ubers, and lyfts at that specific latitude and longitude coordinate pair. This part of the analysis was very difficult since I could not figure out an efficient process for determining the number of crimes, ubers, and lyfts at a specific location since some locations had no crimes or no rideshares. I settled on utilizing another .apply with a lambda function to iterate through every location pair and return the row size of each data frame sliced to only include that location pair.
# 

# %%
### Joining Data to create a comprehensive list of everything that happens at each lat and lon
loc = uber_data.loc[:, ['Lat','Lon']]
loc2 = lyft_data.loc[:, ['Lat', 'Lon']]
loc3 = crime_data.loc[:, ['Lat', 'Lon']]

# %%
locations = pd.concat([loc, loc2, loc3])
locations.value_counts()

# %%
locations.drop_duplicates(inplace=True)
locations.value_counts()

# %%
## Helper Functions
def num_crime(x, y):
    return crime_data.loc[(crime_data['Lat'] == x) & (crime_data['Lon'] == y), :].shape[0]
    
def num_uber(x, y):
    return uber_data.loc[(uber_data['Lat'] == x) & (uber_data['Lon'] == y), :].shape[0]

def num_lyft(x, y):
    return lyft_data.loc[(lyft_data['Lat'] == x) & (lyft_data['Lon'] == y), :].shape[0]

# %%
locations['num_crime'] = locations.apply(lambda x: num_crime(x['Lat'], x['Lon']), axis=1)
locations['num_lyft'] = locations.apply(lambda x: num_lyft(x['Lat'], x['Lon']), axis=1)
locations['num_uber'] = locations.apply(lambda x: num_uber(x['Lat'], x['Lon']), axis=1)

# %%
locations.head()

# %% [markdown]
# Once the data was compiled together, we ran a simple .corrwith the number of crimes and found a negligible correlation between Uber and crime but a weak positive correlation between lyft and crime. While these correlation values (0.1516 and 0.2384) are both relatively negligible, the difference between them sparked us to explore if there was any meaningful distinction between the correlation values within each borough. 
# 
# The process of defining each location within a borough was at first, very daunting. I anticipated the borough defining would be problematic but I was ultimately able to circumvent the issue by utilizing information from the uncleaned crime dataset. The uncleaned crime dataset actually includes the borough that each crime occurred in, through yet another .apply and lambda function, I was able to define each latitude and longitude pair to a borough. 

# %%
locations.corrwith(locations.num_crime).sort_values(ascending = False)

# %%
### Borough v2
def borough2(x, y):
    if crime_data.loc[(crime_data['Lat']==x) & (crime_data['Lon']),:].shape[0] == 0:
        return 'N/A'
    else:
        return crime_data.loc[(crime_data['Lat']==x) & (crime_data['Lon']),:].iloc[0]['borough']
    
locations['Borough'] = locations.apply(lambda x: borough2(x['Lat'], x['Lon']), axis=1)

locations_dummies = pd.get_dummies(locations['Borough'])
locations = pd.concat([locations, locations_dummies], axis=1)
locations.head()

# %% [markdown]
# After the boroughs were assigned for each location, we ran another simple .corrwith the number of crimes and found that the actual borough location had no strong correlation with the number of crimes that occurred at that location.

# %%
locations[['num_crime', 'BRONX', 'BROOKLYN', 'MANHATTAN', 'N/A', 'QUEENS', 'STATEN ISLAND']].corrwith(locations.num_crime).sort_values(ascending = False)

# %%
locations[['num_lyft', 'BRONX', 'BROOKLYN', 'MANHATTAN', 'N/A', 'QUEENS', 'STATEN ISLAND']].corrwith(locations.num_lyft).sort_values(ascending = False)

# %%
locations[['num_uber', 'BRONX', 'BROOKLYN', 'MANHATTAN', 'N/A', 'QUEENS', 'STATEN ISLAND']].corrwith(locations.num_uber).sort_values(ascending = False)

# %%
### Create Heatmap
loc_heat = locations.drop(['Lat', 'Lon', 'N/A'], axis=1)
sns.heatmap(loc_heat.corr(numeric_only=True))

# %%
### Create a pairplot
loc_heat = loc_heat[~loc_heat.index.duplicated()]
loc_heat = loc_heat.drop(['MANHATTAN', 'BRONX', 'BROOKLYN', 'QUEENS', 'STATEN ISLAND'], axis=1)
sns.pairplot(loc_heat, hue='Borough', kind='reg', plot_kws= dict(scatter_kws= dict(s=4)))

# %% [markdown]
# We grouped each location by the borough that it occurred in, then found the correlation values between uber, lyft, and crime for each group of locations. This was likely the most interesting part of the analysis. For every Borough, Lyft has a stronger correlation with crime density than uber. Queens and Manhattan specifically are close to even sharing a positive correlation value. What this data primarily shows, is that in Queens and Manhattan, rideshare is more strongly correlated with crime (likely due to their population densities or some other confounding variable) and lyft has a stronger correlation with crime density than Uber in every borough of New York City. However, as a note, all of these values are fairly low, either indicating a weak positive correlation or a negligible one. This data was then presented in a Facetgrid separated by borough, as well as a simple regplot for both uber/crime and lyft/crime.

# %%
### Check correlation per Borough: Uber -> Crime
locations.groupby('Borough').apply(lambda x: x['num_uber'].corr(x['num_crime'])).sort_values(ascending=False)

# %%
### Check correlation per Borough: Lyft -> Crime
locations.groupby('Borough').apply(lambda x: x['num_lyft'].corr(x['num_crime'])).sort_values(ascending=False)

# %%
### Remove outliers 
no_outliers = locations.loc[(locations['num_crime']<2500) & (locations['num_lyft']<350) & (locations['num_uber']<15000)]

### Grid comparing correlation values grouped by Borough
g = sns.FacetGrid(data=no_outliers, col='Borough', hue='Borough', col_wrap=3, col_order=['QUEENS', 'MANHATTAN', 'BROOKLYN', 'BRONX', 'STATEN ISLAND'])
g.map(sns.regplot,'num_lyft','num_crime', ci=95)

# %%
ax = sns.lmplot(data=no_outliers, x='num_lyft', y='num_crime')
ax.set(xlabel='Number of Lyfts', ylabel='Number of Crimes', title='Relationship between Lyfts and Crime in NYC')
plt.show()

# %%
ax = sns.lmplot(data=no_outliers, x='num_uber', y='num_crime')
ax.set(xlabel='Number of Ubers', ylabel='Number of Crimes', title='Relationship between Ubers and Crime in NYC')

# %% [markdown]
# ## Conclusions
# From the onset, the aim of this analysis was to determine the relationship, if any, between rideshare usage and crime rates in New York City. To begin, the first analysis examined rideshare usage in each borough, in which it was determined that, by a large margin, the borough of Manhattan has the highest concentration of Uber rides called. Although this data only consists of portions of 2014 and 2015, the margin in which Manhattan dominates the percentage of Ubers called allows us to confidently assume that more generally, this borough is the main hotspot for Ubers. Logically this makes sense, as the business and cultural center of New York City (downtown New York City)  resides in this borough, causing one to suspect that the large majority of traffic in the city would be to and from this area. The second analysis visualized the steady growth that rideshare companies have had since 2015, and reflects their dominance over traditional VFH companies. Furthermore, the two bar plots displaying Uber and crime complaint counts show how the two have similar rates of occurrence on an hour-to-hour basis. These portions of analyses 1 and 2 reveal the large scale at which the ridesharing industry operates and its influence over the entire transportation landscape in New York City, as well as the similarities ridesharing rates and crime rates share throughout the course of the day. Thus, the group’s initial desire to determine how ridesharing and crime is justified.
# 
# Next, Crime density by borough was analyzed, and it was found that Manhattan has the highest crime density, while Staten Island has the lowest. Combining this finding with the information from analysis 1, we can conclude that, comparatively, Manhattan is the area of highest crime density as well as the area of highest Uber ride activity. Thus, the relationship between crime density  rates and uber activity is that both have their highest rates in the borough of Manhattan. It should be noted that the relationship does not suggest that the high Uber ride activity is leading to high crime density in this area, nor vice versa. Nonetheless, this relationship exists and could prove useful for groups looking to improve the ridesharing experience in New York City.
# 
# Finally, Uber, Lyft, and crime correlations were computed for each of the boroughs. Despite all of the correlations being fairly low, the notable observations from this computation were that ridesharing in the boroughs of Queens and Manhattan have stronger correlations with crime than any of the other boroughs (the reason for this is not clear), and that Lyft has a stronger correlation with crime density than Uber in all of the boroughs. Together, these 4 analyses allow us to conclude that crime and rideshare activity share similar occurrence rate patterns, Manhattan has both the highest density of crime rates and the highest density of rideshare activity, ridesharing in Queens and Manhattan are most correlated with crime rates, and that Lyft has a higher correlation with crime density than Uber.

# %% [markdown]
# ## Recommendations to stakeholder(s)
# Based on our analysis, we recommend that rideshare companies focus their marketing campaigns in Manhattan. There is already a market for ridesharing in Manhattan shown by the fact that it is used more in Manhattan than any of the other boroughs. While ridesharing is already popular during rush hour, rideshare companies could emphasize the statistic that crime is higher in the evening. They could encourage people to use ridesharing more at night to ensure safe travel. On top of that, we recommend that rideshare companies focus on improving pickup efficiency in areas with more crime to help users feel safer.
# 
# We recommend that the NYPD focuses its crime fighting efforts on areas in Manhattan where ridesharing pick-up and drop-offs are high to make transportation safer. Making popular pick-up and drop-off areas safer would be beneficial to the City of New York because people would feel more comfortable going out. If people are going out more and spending more money, it would benefit the city’s economy.
# 
# We recommend that rideshare users be aware of the areas with higher reported crime densities, especially in Manhattan during rush hour. Choose pick-up and drop-off locations that have lower reported crime densities when possible.
# 
# It is important to note that our data is not very highly correlated, so it may be difficult for the stakeholders to see notable results if they follow our recommendations. To implement our recommendations in a more specific way, stakeholders could look at smaller groupings of areas in New York that they are interested in. This would likely provide more accurate results.

# %% [markdown]
# ## References
# [1] NYC Taxi and Limo Commission. (2018, March 21). Taxi Zone Lookup. data.world. Retrieved December 5, 2022, from https://data.world/nyc-taxi-limo/taxi-zone-lookup 
# 
# [2] Burgueño Salas, E. (2022, April 20). Ridesharing Services in the U.S. Statista. Retrieved December 2, 2022, from https://www.statista.com/topics/4610/ridesharing-services-in-the-us/#topicOverview
# 
# [3] Burgess, M. (2011, April). Understanding Crime Hotspot Maps. NSW Bureau of Crime Statistics and Research. Retrieved December 4, 2022, from https://www.bocsar.nsw.gov.au/Publications/BB/bb60.pdf 
# 
# [4] Plotly. (2022). Mapbox Density Heatmap in Python. Plotly Graphing Libraries. Retrieved December 2, 2022, from https://plotly.com/python/mapbox-density-heatmaps/
# 
# [5] Mango, T. (2019, July 16). Plotting GeoJson files with Matplotlib. Medium. Retrieved December 2, 2022, from https://medium.com/@tmango/plotting-geojson-files-with-matplotlib-5ed87df570ab
# 
# [6] Raymond, R. (2021, August 22). Adding Geopandas Boundary Plot to Plotly. Stack Overflow. Retrieved December 2, 2022, from https://stackoverflow.com/questions/68880787/adding-geopandas-boundary-plot-to-plotly 
# 
# [7] U.S. Census Bureau. (n.d.). 2020 Census Demographic Data Map Viewer. A Story Map. Retrieved December 4, 2022, from https://mtgis-portal.geo.census.gov/arcgis/apps/MapSeries/index.html?appid=2566121a73de463995ed2b2fd7ff6eb7


