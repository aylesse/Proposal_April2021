# Proposal_April2021
Code for Capstone Proposal April 2021

Increasing the use of electric vehicles (EV) while decreasing the use of cars that run on fossil fuels has the potential to combat climate change and is a major goal of the Biden Administration. In addition to the economic hurdles that must be overcome to reach this goal in the United States, a major obstacle is the requirement of national EV charging station infrastructure. Optimal EV charging station placement is dependent on variables such as available green energy sources, traffic flow, parking space, etc.

I propose a capstone project that will use publicly available data from sources such as The Federal Highway Administration, The U.S. Department of Transportation, The U.S. Wind Turbine Database, and The National Renewable Energy Laboratory, as well as highway locations and information on EV charging times and EV battery life, to predict the optimal locations of EV charging stations. Depending on the ultimate direction and scope of this project, data pertaining to the weather and satellite images may also be of use.

My exploratory analysis showed that Texas increased its highway use of gasoline from 2000-2018 and is second in the country with regards to highway use of gasoline as of 2018, with only California exceeding it. However, the number of EV charging stations in California, which has not increased its highway use of gasoline in recent years, vastly exceeds that of Texas. I have also found that Texas leads the country in wind turbines, suggesting that it has the capacity to increase EV charging stations while powering them with green energy. Thus, I will focus my initial analyses and predictions to find the optimal EV charging station locations in Texas.

To demonstrate the viability of this project I analyzed the locations (longitude and latitude data) of all the publicly available EV charging stations in Texas, the locations of all of the wind turbines in Texas, and a small data set of locations of truck stop parking obtained from the Bureau of Transportation. Plotting these locations, I found that dense clusters of current EV charging stations do not overlap with wind turbine clusters (current EV charging stations appear to be concentrated near major cities) and that truck stop parking is disperse throughout that state.

I first performed DBSCAN clustering on the locations of the EV charging stations in Texas. I used this clustering to separate the densely clustered EV charging stations from the disperse EV charging stations. 

I then set this disperse set of EV charging stations as the target of a linear regression model to predict locations separate from those already heavily populated by current EV charging stations. I trained the model on randomized subsets of the locations of the wind turbines and the truck stops. This model predicted new locations that are closer to the wind turbines and do not overlap with the current dense clusters of EV charging stations.  While this initial model is limited by the small number of truck stops available for training data, these preliminary analyses suggest that a richer training dataset that includes more information on highway locations, battery life, etc., could be used to predict optimal EV charging station locations.

Links to data sources are provided in the comments.
