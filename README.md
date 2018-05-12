# BigML

It is the realization for https://papers.nips.cc/paper/6160-temporal-regularized-matrix-factorization-for-high-dimensional-time-series-prediction
Additional things are script for collecting data from hitbtc (for experiments with data with small timestep), simplest trading bots, reinforcement learning bots(in future)

# Preparing of shad datasets 
First of all we decided to try to learn only on the time period, on which data for all currencies is provided. We found this intersections and it appears that it is empty. Time period means continuous segment from the minimal timestamp of some currency to maximal. Finding that the intersection of such segments for completely all currencies is empty we decided to analyze it deeper. We solved the following problem - for every given k find the the maximal intersection with at least k segments. After that we plotted graphs for this maximal intersections from k. (you can find them and full analyzis in the analyzing dataset.ipynb). It appears that for one, five, thirty minuts and hours data we cal take almost all currencies and for them intersection will be very huge. For the day's data unfortunately situation is wrong. 

So for all data except days we prepared two datasets - of such huge intersections, and fulls. In the first case the duration is lower, but the number of missing values is also small. In the second case duration is bigger, but at the price of huge amount of missing values.

#Collecting hitbtc datasets
We collected some data from the hitbtc. For every currencies pair we have some time series, with frequency about 1 measurement per second, but unfortunately the times between neibour measurements are not exactly one second, but some value close to it. Also sometimes there are huge delays with no data for some very huge intervals like 10-20 seconds. We decided to treat it this way - we converted this to equidistant dataset with delay equal 100 seconds. In the case when all is normal we use linear interpolation, in the case of mentioned earlier missing intervals we put to such places missing values. Eventually hitbtc dataset consist of 3 datasets for btc, usd and eth markets, with common timestamps and one joint dataset with this common timestamps.

