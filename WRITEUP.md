# Write-Up

## Approach

I approached the problem as a anomaly detection problem as the machines which have worn out would behave significantly different than if they are fine. I decided to built a primary and secondary anomaly detector. The primary detector is regression based detector where I try to fit a linear regression model to predict peak vibration given power, machine type, part type and then use the residual between the predicted and actual vibration values to flag anomalous i.e. worn out machines using median/MAD based on 6 sigma cutoff. Motivation behind chosing a linear regressor is straight forward. 
A Linear Model 
1. Directly represents the physical relationship.
2. Does not flag a high-power job merely because vibration is also high.
3. Interpretable with a small dataset.
4. Produces a clear anomaly score in vibration units.

I evaluate different version of regression such as ridge, Huber and robust and select the one which has the best fit on the held out data set. 

Since there are no ground truth labels, I decide to have a secondary detector which sort of acts as validator for the primary detector. for the secondary detector I use Isolation forest, Local Outlier factor and Mahalanobis distance. Secondary detector is more like a classifier which uses the labels from primary detector as ground truth.

I flag every reading of the sensor as normal or anomaly. A reading-level flag is noisy but the task is to flag for the jon. A job is flagged only when enough of its readings are flagged by **both** stages — sustained chatter, not one transient. Jobs with too few cutting readings get *no verdict* rather than a guess.


## Data exploration
Goal for data exploration was to see for any data sanity issues in the individual raw csv files. One important observation was the discrepancy in the time zone between sensor data and log data. This is very important for the aggregation of all three data sources. Rather than *assuming* a timezone, we estimate the offset from the physics: **cutting draws power and idling does not**, so the correct offset is the one that makes MES job windows coincide with the high-power stretches. We scan candidate offsets and maximise `mean(power inside jobs) - mean(power outside jobs)`

Another important observation from stand alone analysis of the raw sensor files was Power is a clean 1 Hz with no gaps. Vibration is 2 Hz with **two dropouts** (20 min on one machine, 12 min on another). Consequence: some jobs will have less vibration coverage than others, so per-job verdicts must account for how many readings actually back them up rather than treating every job as equally well observed.

Also, from EDA we observed that the Vibration scales multiplicatively with power. Hence we model Log(Vibration) vs log(Power) 

## Validation

To check for correction of the results, I used a secondary anomaly detector which would check if the primary anaomaly detector is flagging the right jobs

## Tool-wear findings

Which jobs show signs of tool wear? How confident are you?

| Rank | Job | Machine | Part | Mean power (kW) | Peak vib (g) | Wear score (σ) |
|---|---|---|---|---|---|---|
| 1 | JOB-120 | CNC-09 | AL-panel | 5.33 | 3.14 | 9.63 |
| 2 | JOB-104 | CNC-09 | TI-fitting | 14.15 | 6.49 | 9.59 |
| 3 | JOB-206 | CNC-11 | ST-housing | 10.59 | 4.99 | 9.54 |
| 4 | JOB-024 | CNC-07 | ST-gear | 9.26 | 6.93 | 9.26 |
| 5 | JOB-224 | CNC-11 | AL-panel | 5.50 | 3.26 | 9.05 |
| 6 | JOB-210 | CNC-11 | AL-bracket | 3.77 | 2.36 | 8.80 |
| 7 | JOB-007 | CNC-07 | ST-housing | 10.70 | 7.58 | 8.63 |
| 8 | JOB-106 | CNC-09 | AL-bracket | 3.86 | 2.42 | 8.55 |
| 9 | JOB-018 | CNC-07 | AL-bracket | 4.21 | 3.18 | 7.12 |
| 10 | JOB-032 | CNC-07 | ST-housing | 10.75 | 6.92 | 6.78 |
| 11 | JOB-040 | CNC-07 | ST-housing | 10.71 | 7.02 | 6.50 |
| 12 | *JOB-212* | *CNC-11* | *ST-gear* | *8.02* | *1.84* | *2.72 — not flagged* |

## What I'd do differently

If I had more time and had data for more jobs, I would have tried more feature engineering to learn more spectral features from time domain sensor data and then have a one feature set for every job rather than flag every reading which is very noisy. Also I would have definitely time series based Auto encoder approach if I had more data. Alternatively with less data, I could also have used zero shot classification capabilities of tabular foundational models.
 