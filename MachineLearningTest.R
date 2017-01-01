# Correlation One Machine Learning Test
# Joe Silverstein

library(data.table)
library(quantregForest)
library(zoo)

set.seed(13)

dt = fread("./data/stock_returns_base150.csv")

pastdt = dt[1:50,]
futuredt = dt[51:100,]

X = pastdt[, 3:11, with = FALSE]
Y = unlist(pastdt[, 2, with = FALSE])
qrf = quantregForest(X, Y, keep.inbag = TRUE)
pred = predict(qrf, what = 0.5)
mae0 = mean(abs(pred - Y)) # OOB MAE

# Tried LAD gradient boosting with MAE split criterion in Python, but it doesn't perform
# nearly as well. That might be because I need to optimize the parameters, but doing that
# along with optimizing the features used will be very computationally expensive, in addition
# to forcing me to use a test set because there is no OOB prediction. It's 
# better to just use quantile random forest because of the OOB predictions, the 
# semi-automatic feature selection, and the importance measure.

# Now add the lagged values of S1,...,S10. Since random forests have somewhat automatic 
# feature selection, the only problem is dealing with the missing values. Once one of the
# lagged values is missing, I have to delete all the other lags for that date as well. So
# just test the model with one lag for all the variables, two lags, three lags, etc. to try
# to minimize the OOB MAE.

data = zoo(dt)

# lagdtlist = vector("list", 21)
# lagdtlist[[1]] = dt
# for (i in 1:5) {
#   lagdtlist[[i+1]] = data.table(cbind(unlist(lagdtlist[[i]]), lag(data[, 2:11], -i, na.pad = TRUE)))
# }

# The loop above won't finish because R sucks at loops. Instead, do it without loops.
# lagdtlist = vector("list", 11)
# lagdtlist[[1]] = dt
# i = 1
# lagdtlist[[i+1]] = data.table(cbind(lagdtlist[[i]], lag(data[, 2:11], -i, na.pad = TRUE)))

# Can't use lagged values of S1 because they don't exist, and predicting them to use as features
# would introduce a huge amount of additional inaccuracy. should only use lagged values of the 
# other variables.

# Just add the lagged variables by hand
lag1 = lag(data[, 3:11], -1, na.pad = TRUE)
lag2 = lag(data[, 3:11], -2, na.pad = TRUE)
lag3 = lag(data[, 3:11], -3, na.pad = TRUE)
lag4 = lag(data[, 3:11], -4, na.pad = TRUE)
lag5 = lag(data[, 3:11], -5, na.pad = TRUE)
lag6 = lag(data[, 3:11], -6, na.pad = TRUE)
lag7 = lag(data[, 3:11], -7, na.pad = TRUE)
lag8 = lag(data[, 3:11], -8, na.pad = TRUE)
lag9 = lag(data[, 3:11], -9, na.pad = TRUE)
lag10 = lag(data[, 3:11], -10, na.pad = TRUE)

lag1dt = data.table(cbind(data, lag1))
lag2dt = data.table(cbind(data, lag1, lag2))
lag3dt = data.table(cbind(data, lag1, lag2, lag3))
lag4dt = data.table(cbind(data, lag1, lag2, lag3, lag4))
lag5dt = data.table(cbind(data, lag1, lag2, lag3, lag4, lag5))
lag6dt = data.table(cbind(data, lag1, lag2, lag3, lag4, lag5, lag6))
lag7dt = data.table(cbind(data, lag1, lag2, lag3, lag4, lag5, lag6, lag7))
lag8dt = data.table(cbind(data, lag1, lag2, lag3, lag4, lag5, lag6, lag7, lag8))
lag9dt = data.table(cbind(data, lag1, lag2, lag3, lag4, lag5, lag6, lag7, lag8, lag9))
lag10dt = data.table(cbind(data, lag1, lag2, lag3, lag4, lag5, lag6, lag7, lag8, lag9, lag10))

## Build quantile random forest and test OOB error for each number of lags

mae = numeric(11)
mae[1] = mae0
mae

completelag1dt = lag1dt[complete.cases(lag1dt), ]
completelag1dt = sapply(completelag1dt, function(x) as.numeric(as.character(x)))
X = completelag1dt[, 3:ncol(completelag1dt)]
Y = unlist(completelag1dt[, 2])
qrf = quantregForest(X, Y, keep.inbag = TRUE)
pred = predict(qrf, what = 0.5)
mae[2] = mean(abs(pred - Y))
mae

completelag2dt = lag2dt[complete.cases(lag2dt), ]
completelag2dt = sapply(completelag2dt, function(x) as.numeric(as.character(x)))
X = completelag2dt[, 3:ncol(completelag2dt)]
Y = unlist(completelag2dt[, 2])
qrf = quantregForest(X, Y, keep.inbag = TRUE)
pred = predict(qrf, what = 0.5)
mae[3] = mean(abs(pred - Y))
mae

completelag3dt = lag3dt[complete.cases(lag3dt), ]
completelag3dt = sapply(completelag3dt, function(x) as.numeric(as.character(x)))
X = completelag3dt[, 3:ncol(completelag3dt)]
Y = unlist(completelag3dt[, 2])
qrf = quantregForest(X, Y, keep.inbag = TRUE)
pred = predict(qrf, what = 0.5)
mae[4] = mean(abs(pred - Y))
mae

completelag4dt = lag4dt[complete.cases(lag4dt), ]
completelag4dt = sapply(completelag4dt, function(x) as.numeric(as.character(x)))
X = completelag4dt[, 3:ncol(completelag4dt)]
Y = unlist(completelag4dt[, 2])
qrf = quantregForest(X, Y, keep.inbag = TRUE)
pred = predict(qrf, what = 0.5)
mae[5] = mean(abs(pred - Y))
mae

completelag5dt = lag5dt[complete.cases(lag5dt), ]
completelag5dt = sapply(completelag5dt, function(x) as.numeric(as.character(x)))
X = completelag5dt[, 3:ncol(completelag5dt)]
Y = unlist(completelag5dt[, 2])
qrf = quantregForest(X, Y, keep.inbag = TRUE)
pred = predict(qrf, what = 0.5)
mae[6] = mean(abs(pred - Y)) 
mae # 5th lag seems important

completelag6dt = lag6dt[complete.cases(lag6dt), ]
completelag6dt = sapply(completelag6dt, function(x) as.numeric(as.character(x)))
X = completelag6dt[, 3:ncol(completelag6dt)]
Y = unlist(completelag6dt[, 2])
qrf = quantregForest(X, Y, keep.inbag = TRUE)
pred = predict(qrf, what = 0.5)
mae[7] = mean(abs(pred - Y)) 
mae

completelag7dt = lag7dt[complete.cases(lag7dt), ]
completelag7dt = sapply(completelag7dt, function(x) as.numeric(as.character(x)))
X = completelag7dt[, 3:ncol(completelag7dt)]
Y = unlist(completelag7dt[, 2])
qrf = quantregForest(X, Y, keep.inbag = TRUE)
pred = predict(qrf, what = 0.5)
mae[8] = mean(abs(pred - Y)) 
mae

completelag8dt = lag8dt[complete.cases(lag8dt), ]
completelag8dt = sapply(completelag8dt, function(x) as.numeric(as.character(x)))
X = completelag8dt[, 3:ncol(completelag8dt)]
Y = unlist(completelag8dt[, 2])
qrf = quantregForest(X, Y, keep.inbag = TRUE)
pred = predict(qrf, what = 0.5)
mae[9] = mean(abs(pred - Y)) 
mae # 8th lag also important

completelag9dt = lag9dt[complete.cases(lag9dt), ]
completelag9dt = sapply(completelag9dt, function(x) as.numeric(as.character(x)))
X = completelag9dt[, 3:ncol(completelag9dt)]
Y = unlist(completelag9dt[, 2])
qrf = quantregForest(X, Y, keep.inbag = TRUE)
pred = predict(qrf, what = 0.5)
mae[10] = mean(abs(pred - Y)) 
mae

completelag10dt = lag10dt[complete.cases(lag10dt), ]
completelag10dt = sapply(completelag10dt, function(x) as.numeric(as.character(x)))
X = completelag10dt[, 3:ncol(completelag10dt)]
Y = unlist(completelag10dt[, 2])
qrf = quantregForest(X, Y, keep.inbag = TRUE)
pred = predict(qrf, what = 0.5)
mae[11] = mean(abs(pred - Y)) 
mae

# If I add more lags, I'll be running short on data to use.

# Try constructing dataset with lags 1, 2, and 5, since those seem to be the important ones:
customlagdt = data.table(cbind(data, lag1, lag2, lag5))
completecustomlagdt = customlagdt[complete.cases(customlagdt), ]
completecustomlagdt = sapply(completecustomlagdt, function(x) as.numeric(as.character(x)))
X = completecustomlagdt[, 3:ncol(completecustomlagdt)]
Y = unlist(completecustomlagdt[, 2])
qrf = quantregForest(X, Y, keep.inbag = TRUE)
pred = predict(qrf, what = c(0.025, .5, 0.975))
(custom_mae = mean(abs(pred[, 2] - Y))) # Best so far. Use this one.

# # For simplicity, just use all lags up to 8, since that was the best MAE. 
# completelag8dt = lag8dt[complete.cases(lag8dt), ]
# completelag8dt = sapply(completelag8dt, function(x) as.numeric(as.character(x)))
# X = completelag8dt[, 3:ncol(completelag8dt)]
# Y = unlist(completelag8dt[, 2])
# qrf = quantregForest(X, Y, keep.inbag = TRUE)
# pred = predict(qrf, what = c(0.025, 0.5, 0.975))
# final_mae = mean(abs(pred[, 2] - Y))

# Determine how often the direction of the change is significant (nonzero with >95% prob)
significant = pred[, 1] * pred[, 3] > 0
sum(significant) / length(significant) 
# At least 42% of the time, the direction of the daily change can be predicted with 95% confidence

# Sort variables by importance
featureImportance = importance(qrf)
featureImportance = data.table(rownames(featureImportance), featureImportance)
(featureImportance = featureImportance[order(featureImportance$IncNodePurity, decreasing = TRUE),])

futurepreds = predict(qrf, newdata = as.matrix(customlagdt[51:100, 3:ncol(customlagdt), with = FALSE]))

# How often are the future value predictions significant?:
significant = futurepreds[, 1] * futurepreds[, 3] > 0
sum(significant) / length(significant)
# 62% are significant at the 95% confidence level!

submission = cbind(customlagdt[51:100, 1, with = FALSE], futurepreds[, 2])
names(submission) = c("date", "Value")
sum(submission$Value) # 1.8
write.csv(submission, file = "predictions.csv")
