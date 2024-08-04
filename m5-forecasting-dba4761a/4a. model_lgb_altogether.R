#--- Read in "training.rds" for the training dataset singular
training <- readRDS(file.path(path_data_clean, "training.rds"))

# Here, we modify the dataset for model training
y <- training$sales # Extracting sales numbers for model training as the dependent variable

# Obtaining the indices of the training set minus the most recent 28*2=56 days
x_id <- training[date <= max(date)-2*h, which = TRUE] 

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

#--- LGB Model 3 50 mins to run model, 3hrs with regularisation grid 
params <- list(
  boosting_type = "gbdt",
  objective = "tweedie", #sales data contains many zero counts, which can be modelled well with the Tweedie distribution 
  tweedie_variance_power = 1.1,
  metric = "rmse", #we want to minimise rmse as our objective
  subsample = 0.5, #subsampling helps to reduce training time without substantially compromising the learning process
  subsample_freq = 1, #prevent overfitting since the model sees a different subset of data at each step
  learning_rate = 0.03, #small enough to be able to achieve optimality without, large enough to reduce computing time
  num_leaves = 2047,
  min_data_in_leaf = 4095, # large minimum data in leaf helps in controlling overfitting
  feature_fraction = 0.5,
  max_bin = 100,
  n_estimators = 1400,
  boost_from_average = FALSE #target variable has many zeros, prevents the model from being biased towards the mean sales
  )

lambda_l1_grid <- seq(0.20, 0.80, by = 0.20) #regularisation grid 
best_score <- Inf  
best_lambda_l1 <- NULL
best_model <- NULL

tic() # 3 hours
for (i in lambda_l1_grid) {
  params$lambda_l1 <- i
  m_lgb_3 <- lgb.train(params = params,
                       data = x_train,
                       nrounds = 500,
                       valids = list(train = x_train, valid = x_val),
                       early_stopping_rounds = 300,
                       eval_freq = 100)
  
  if (m_lgb_3$best_score < best_score) {
    best_score <- m_lgb_3$best_score
    best_lambda_l1 <- i
    best_model <- m_lgb_3
  }
}
toc()
cat("Best score:", m_lgb_3$best_score, "at", m_lgb_3$best_iter, "iteration") #1.799

lgb.plot.importance(lgb.importance(m_lgb_3), 15)

#--- Save the model
#saveRDS.lgb.Booster(m_lgb_3, "m_lgb_final.rds")

#--- Read in lightGBM model
#m_lgb_3 <- readRDS.lgb.Booster(file.path(path_model, "m_lgb_final.rds"))

rm(x_train, x_val, params)
free()

########## MAKING PREDICTIONS ##########

#--- Creating our forecasts for LGB Singular
te <- create_dt(FALSE) #preparing test set

macroeconomic_data <- fread(file.path(path_data_raw, "MacroeconomicFeatures.csv"))
macroeconomic_data$year <- as.integer(macroeconomic_data$year)
macroeconomic_data$month <- as.integer(macroeconomic_data$month)
macroeconomic_data$quarter <- as.integer(macroeconomic_data$quarter)

tic() 
# here, we make predictions for d_1914 to d_1969 (56 days in total)
# d_1914 to d_1941 is for validation
# d_1942 to d_1969 is for evaluation
for (day in as.list(seq(fday, length.out = 2*h, by = "day"))){
  cat(as.character(day), " ")
  tst <- te[date >= day - max_lags & date <= day]
  create_fea_lgb_singular(tst)
  tst <- merge(tst, macroeconomic_data, by = c("year", "month", "quarter"), all.x = TRUE)
  tst <- data.matrix(tst[date == day][, c("id", "sales", "date") := NULL])
  te[date == day, sales := predict(m_lgb_3, tst)]}
toc()

#--- Save the predictions made as RDS
#saveRDS(te, file = "predictions_lgb.rds")

#--- Load in the predictions made for LGB Singular
#te <- readRDS(file.path(path_data_clean, "predictions_lgb.rds"))

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

sqrt(mean((my_dataframe$actual - my_dataframe$pred)^2)) #2.34

rm(list=c("actual_val", "my_pred", "my_dataframe"))

########## GENERATE OUTPUT ##########

#--- Preparing output file
te[date >= fday #retain only the rows where the date is on or after "2016-04-25"
][date >= fday+h, id := sub("validation", "evaluation", id) #if the date is on or after fday + h (where h is 28), "validation" is substituted with "evaluation" in the id field.
][, d := paste0("F", 1:28), by = id #new column d is created with values "F1", ..., "F28" grouped by id.
][, dcast(.SD, id ~ d, value.var = "sales") #reshape the data table from a long format to a wide format
][, fwrite(.SD, "final_output_lgb.csv")] #write output file to workflow folder

#--- Finally, clear the environment
rm(list=ls())
