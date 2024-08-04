#--- Read in "training.rds" for the training dataset
training <- readRDS(file.path(path_data_clean, "training_rf.rds"))

# Here, we modify the dataset for model training
y <- training$sales # Extracting sales numbers for model training as the dependent variable

# Obtaining the indices of the training set minus the most recent 28 days
x_id <- training[date <= max(date) - h, which = TRUE] 
print(paste("Rows in training set:", length(x_id)))

training[, c("id", "sales", "date") := NULL] # Removing id, sales, date
free()

training <- data.matrix(training) # Converting to a matrix for training
free()

categorical_v <- c("item_id", "state_id", "dept_id", "cat_id",
                   "wday", "mday", "week", "month", "quarter", "year", "is_weekend",
                   "snap_CA", "snap_TX", "snap_WI")

x_train <- lgb.Dataset(training[x_id, ], label = y[x_id], categorical_feature = categorical_v) 
x_val <- lgb.Dataset(training[-x_id, ], label = y[-x_id], categorical_feature = categorical_v)

rm(training, y, x_id)
free()
rm("macroeconomic_data")

########## MODEL TRAINING ##########

#--- Seed for Reproducibility
set.seed(4761)

# Set parameters for LightGBM Random Forest
params <- list(
  objective = "regression", # predict continuous values 
  boosting = "rf", # using random forest 
  bagging_freq = 1, # bagging shld be performed at each iteration to increase randomness
  bagging_fraction = 0.8, # prevent overfitting
  feature_fraction = 0.8, # feature sampling in a usual random forest 
  num_trees = 600, # larger number of trees bc of 11 mil rows, more stable and accurate but longer
  max_depth = -1 #let trees grow and capture complex patterns
)

# Train the model
lgb_rf_model <- lgb.train(
  params = params,
  data = x_train,
  valids = list(train = x_train, valid = x_val),
  num_boost_round = 100,
  verbose = 1
)

cat("Best score:", lgb_rf_model$best_score, "at", lgb_rf_model$best_iter, "iteration") 

#--- Save the model
#saveRDS.lgb.Booster(lgb_rf_model, "m_lgb_rf_final.rds")

#--- Read in lightGBM model
#lgb_rf_model <- readRDS.lgb.Booster(file.path(path_model, "m_lgb_rf_final.rds"))

lgb.plot.importance(lgb.importance(lgb_rf_model), 20)

macroeconomic_data <- fread(file.path(path_data_raw, "MacroeconomicFeatures.csv"))
macroeconomic_data$year <- as.integer(macroeconomic_data$year)
macroeconomic_data$month <- as.integer(macroeconomic_data$month)
macroeconomic_data$quarter <- as.integer(macroeconomic_data$quarter)

########## MAKING PREDICTIONS ##########

#--- Creating our forecasts
te <- create_dt(FALSE) #preparing test set

tic()
# 11506.92 sec elapsed
# here, we make predictions for d_1914 to d_1969 (56 days in total)
# d_1914 to d_1941 is for validation
# d_1942 to d_1969 is for evaluation
for (day in as.list(seq(fday, length.out = 2*h, by = "day"))){
  cat(as.character(day), " ")
  tst <- te[date >= day - max_lags & date <= day]
  create_fea_rf(tst)
  tst <- merge(tst, macroeconomic_data, by = c("year", "month", "quarter"), all.x = TRUE)
  tst <- data.matrix(tst[date == day][, c("id", "sales", "date") := NULL])
  te[date == day, sales := predict(lgb_rf_model, tst)]}
toc()

#--- Save the predictions made as RDS
#saveRDS(te, file = "predictions_rf.rds")
#te <- readRDS(file.path(path_data_clean, "predictions_rf.rds"))

free()

########## EVALUATING ##########

#--- Calculating RMSE
# Extract sales predictions from d_1914 to d_1941
my_pred <- te[date >= fday & date < fday+28] %>% select(sales)
actual_val <- fread(file.path(path_data_raw, "sales_train_evaluation.csv"), nrows = Inf) %>% 
  select(-c(7:1919)) %>% 
  melt(measure.vars = patterns("^d_"), variable.name = "d", value.name = "sales") %>%
  select(sales)

my_dataframe <- data.table(my_pred,actual_val)
colnames(my_dataframe) <- c("pred", "actual")

sqrt(mean((my_dataframe$actual - my_dataframe$pred)^2)) 
#rf model:RMSE = 2.441037

rm(list=c("actual_val", "my_pred", "my_dataframe"))

########## GENERATE OUTPUT ##########

#--- Preparing output file
te[date >= fday #retain only the rows where the date is on or after "2016-04-25"
][date >= fday+h, id := sub("validation", "evaluation", id) #if the date is on or after fday + h (where h is 28), "validation" is substituted with "evaluation" in the id field.
][, d := paste0("F", 1:28), by = id #new column d is created with values "F1", ..., "F28" grouped by id.
][, dcast(.SD, id ~ d, value.var = "sales") #reshape the data table from a long format to a wide format
][, fwrite(.SD, "final_output_rf.csv")] #write output file

#--- Finally, clear the environment
rm(list=ls())
