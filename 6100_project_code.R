library(funModeling) 
library(tidyverse)
library(Hmisc)
library(geohashTools)
library(randomForest)
library(class)
library(data.table)
library(kknn)
library(rpart)
library(e1071)
#install.packages("e1071")
library(caret)
require(caTools)

sfc.data <-read.csv(file = 'train.csv')
zip.data <- read.csv(file = 'uszipsv1.4-2.csv')

# Get a glimpse of Data to understad
glimpse(sfc.data)

#Get the metrics about data types, zeros, infinite numbers, and missing values
df_status(sfc.data)

# Analysing the data
describe(sfc.data)

#classifying dates to seasons
fall <- c("09","10","11")
summer <- c("06", "07", "08")
spring <- c("03", "04", "05")
winter <- c("12", "01", "02")
 
sfc.data$season <- apply(sfc.data, 1, FUN = function(x) if(strsplit( x, "-" )[[1]][2] %in% fall) {
   "fall"
 } else if (strsplit( x, "-" )[[1]][2] %in% summer ) {
   "summer"
 } else if (strsplit( x, "-" )[[1]][2] %in% spring ) {
   "spring"
 } else if (strsplit( x, "-" )[[1]][2] %in% winter ) {
   "winter" 
 })

head(sfc.data)

#removing outliers
boxplot(sfc.data$Y)
sfc.data <- sfc.data[!(sfc.data$Y==90),]
boxplot(sfc.data$Y)

#classifying lat,long to geohash
get_geohash <- function(a){
  return(gh_encode(a[9],a[8] , precision = 6L))
}
sfc.data$geohash <- apply(sfc.data, 1, get_geohash)

head(sfc.data)

#classifying crimes 

Theft <- c("LARCENY/THEFT","VEHICLE THEFT","BURGLARY")
Sexual_offences <- c("SEX OFFENSES FORCIBLE", "SEX OFFENSES NON FORCIBLE", "PORNOGRAPHY/OBSCENE MAT","PROSTITUTION")
Public_order <- c("DRUNKENNESS", "SUSPICIOUS OCC", "BRIBERY","DRIVING UNDER THE INFLUENCE","RECOVERED VEHICLE",
                  "BAD CHECKS","LOITERING","DISORDERLY CONDUCT","LIQUOR LAWS","TRESPASS","WEAPON LAWS")
Assault <- c("ROBBERY", "KIDNAPPING", "ASSAULT")
Drug_offences <- c("DRUG/NARCOTIC")
Property_crime <- c("TREA", "EMBEZZLEMENT", "STOLEN PROPERTY","VANDALISM","ARSON")
Whitecollor_crime <- c("FRAUD", "FORGERY/COUNTERFEITING", "SECONDARY CODES")
Victimless_crime <- c("GAMBLING", "RUNAWAY" )
Suicide <- c("SUICIDE", "FAMILY OFFENSES", "MISSING PERSON","EXTORTION")
Other <- c("WARRANTS", "OTHER OFFENSES","NON-CRIMINAL" )

sfc.data$Category <- apply(sfc.data, 1, FUN = function(x) if((x)[2] %in% Theft) {
   "Theft"
} else if ((x)[2] %in% Sexual_offences ) {
   "Sexual offences"
} else if ((x)[2] %in% Public_order ) {
   "Public order"
} else if ((x)[2] %in% Assault ) {
   "Assault"
} else if((x)[2] %in% Drug_offences ) {
   "Drug offences"
} else if ((x)[2] %in% Property_crime ) {
   "Property crime"
} else if((x)[2] %in% Whitecollor_crime ) {
   "White-collor crime"
} else if((x)[2] %in% Victimless_crime ) {
   "Victimless-crime"
} else if ((x)[2] %in% Suicide ) {
   "Suicide"
} else if ((x)[2] %in% Other ) {
   "Other"
})

head(sfc.data)

sapply(sfc.data, function(x) sum(is.na(x))) # reverifying missing values

# Selecting only the zipcodes for the city San Francisco
zip.data <- zip.data[(zip.data$city=='San Francisco'),]
zip.data$geohash <- apply(zip.data, 1, FUN = function(x) gh_encode(x[2],x[3] , precision = 6L))
zip.data = zip.data[c("zip", "lat", "lng", "geohash", "population")]

# Replacing missing values with the mean
zip.data$population[is.na(zip.data$population)] <- round(mean(zip.data$population, na.rm = TRUE))
sapply(zip.data, function(x) sum(is.na(x))) # checking missing values

# innerjoin two data frames by geohash

innerjoin <- inner_join(sfc.data,zip.data,by = "geohash")
head(innerjoin)

df_status(innerjoin)
sanfc.data = innerjoin[!duplicated(innerjoin[c('Dates', 'X', 'Y', 'Descript')]), ]

sanfc.data$hour <- as.numeric(substr(sanfc.data$Dates,12,13))

#Converting to factor variables

sapply(sanfc.data, class)

sanfc.data = sanfc.data[c("Dates", "Category", "Descript", "DayOfWeek", "PdDistrict","Resolution",
                          "Address","season","geohash","zip","lat","lng","population","hour")]

sanfc.data <- transform(sanfc.data,
                        Category=as.factor(Category))
sanfc.data <- transform(sanfc.data,
                        season=as.factor(season))
sanfc.data <- transform(sanfc.data,
                        geohash=as.factor(geohash))

sapply(sanfc.data, function(x) sum(is.na(x))) # checking missing values

# To view the final data set
summary(sanfc.data)

#analyzing numerical variables
plot_num(sanfc.data)

#analyzing categorical variables
freq(sanfc.data)

# 80:20 data
head(sanfc.data)

sample = sample.split(sanfc.data,SplitRatio = 0.80)

sanfc.train <- subset(sanfc.data, sample == TRUE)
head(sanfc.train)

sanfc.test <- subset(sanfc.data, sample == FALSE)

head(sanfc.test)
zip.data = zip.data[c("zip", "lat", "lng", "geohash", "population")]


# Scale dependent variables in 'train'.
x_train_scaled = scale(sanfc.train$lat)
y_train_scaled = scale(sanfc.train$lng)
hour_train_scaled = scale(sanfc.train$hour)

# Scale dependent variables in 'test' using mean and standard deviation derived from scaling variables 
#in 'train'.
x_test_scaled = (sanfc.test$lat - attr(x_train_scaled, 'scaled:center')) / attr(x_train_scaled, 'scaled:scale')
y_test_scaled = (sanfc.test$lng - attr(y_train_scaled, 'scaled:center')) / attr(y_train_scaled, 'scaled:scale')
hour_test_scaled = (sanfc.test$hour - attr(hour_train_scaled, 'scaled:center')) / attr(hour_train_scaled, 'scaled:scale')

days_num = as.numeric(sanfc.data$DayOfWeek)
print(days_num)
pd_num = as.numeric(sanfc.data$PdDistrict)
print(pd_num)
season_num = as.numeric(sanfc.data$season)
print(season_num)
geohash_num = as.numeric(sanfc.data$geohash)
print(geohash_num)

# Create 'train_model' and 'test_model' which only include variables used in the model.
train_model = data.table(category_predict = sanfc.train$Category, 
                         x_scaled = x_train_scaled, 
                         y_scaled = y_train_scaled, 
                         hour_scaled = hour_train_scaled,
                         population = sanfc.train$population,
                         days_num = days_num,
                         pd_num = pd_num,
                         season_num = season_num,
                         geohash_num = geohash_num)

setnames(train_model, 
         names(train_model), 
         c('category_predict', 'x_scaled', 'y_scaled', 'hour_scaled', 'population', 'days_num','pd_num', 'season_num','geohash_num'))

test_model = data.table(x_scaled = x_test_scaled, 
                        y_scaled = y_test_scaled, 
                        hour_scaled = hour_test_scaled)


#####
# CREATE MODEL AND PREDICTIONS.
# Set seed to ensure reproducibility.
set.seed(1)

# Define model.
model = category_predict ~ x_scaled + y_scaled + hour_scaled + population + days_num + pd_num + season_num + geohash_num

model = category_predict ~ x_scaled + y_scaled + hour_scaled


# Create model and generate predictions for training set.
# Variable scaling is done in this command.
knn_train = kknn(formula = model, 
                 train = train_model, 
                 test = train_model, 
                 scale = T)

# Create model and generate predictions for test set.
knn_test = kknn(formula = model, 
                train = train_model,
                test = test_model,
                scale = T)

train_pred = data.table(knn_train$fitted.values)
test_pred = data.table(knn_test$prob)

# View testing accuracy.
print('Testing Accuracy')
print(table(train_model$category_predict == train_pred$V1))
print(prop.table(table(train_model$category_predict == train_pred$V1)))


# Conduct cross validation.
cv = cv.kknn(model, 
             data = train_model, 
             kcv = 2, 
             scale = T)

# View cross validation accuracy.
cv = data.table(cv[[1]])
print('Cross Validation Accuracy')
print(table(cv$y == cv$yhat))
print(prop.table(table(cv$y == cv$yhat)))


# Random Forest
sanfc.rf <- randomForest( sanfc.test$Category ~ sanfc.test$DayOfWeek + sanfc.test$PdDistrict + 
                             sanfc.test$hour ,data = sanfc.test,ntree = 25)
sanfc.rf

#decision tree 
sanfc.dt <- train(Category ~ DayOfWeek + PdDistrict + hour, data = sanfc.train , method = "rpart")
sanfc.dt1 <- predict(sanfc.dt, data = sanfc.train)
table(sanfc.dt1,sanfc.train$Category)
mean(sanfc.dt1 == sanfc.train$Category)

#Cross Validation

sanfc.dtcv <- predict(sanfc.dt, newdata = sanfc.test)
table(sanfc.dtcv,sanfc.test$Category)

mean(sanfc.dtcv == sanfc.test$Category)

