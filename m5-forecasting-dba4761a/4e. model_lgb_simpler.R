#--- Read in "training_lgb_simpler.rds" for the training dataset simpler features
training <- readRDS(file.path(path_data_clean, "training_lgb_simpler.rds"))

y <- training$sales # Extracting sales numbers for model training as the dependent variable

x_id <- training[date <= max(date)-h, which = TRUE] # Obtaining the indices of the training set minus the most recent 28 days

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

########## MODEL TRAINING ##########

#--- Seed for reproducibility
set.seed(4761)

#--- Model Training
params <- list(objective = "poisson", #poisson because suitable for modeling count data, particularly when the counts are small, also computationally less expensive
          metric ="rmse", # our goal to minimise rmse
          force_row_wise = TRUE,
          learning_rate = 0.075,
          sub_feature = 0.8,# select 80% of features before training each tree, speed up and reduce overfitting
          sub_row = 0.75, #similar reasoning as sub_feature
          bagging_freq = 1,
          lambda_l2 = 0.1, #L2 regularisation with low lambda to prevent overfitting as well
          nthread = 4) #2 threads per CPU core

tic() # 82 minutes to train the model
m_lgb <- lgb.train(params = params,
                   data = x_train,
                   nrounds = 2000,
                   valids = list(valid = x_val),
                   early_stopping_rounds = 400,
                   eval_freq = 200)
toc()
cat("Best score:", m_lgb$best_score, "at", m_lgb$best_iter, "iteration") #2.06

#--- Save the model
#saveRDS.lgb.Booster(m_lgb, "m_lgb_simpler.rds")

#--- Read in lightGBM model
#m_lgb <- readRDS.lgb.Booster(file.path(path_model, "m_lgb_simpler.rds"))

lgb.plot.importance(lgb.importance(m_lgb), 10)

rm(x_train, x_val, params)
free()

########## MAKING PREDICTIONS ##########

#--- Creating our forecasts

te <- create_dt(FALSE) #preparing test set

tic() # 150 minutes to make predictions
# here, we make predictions for d_1914 to d_1969 (56 days in total)
# d_1914 to d_1941 is for validation
# d_1942 to d_1969 is for evaluation
for (day in as.list(seq(fday, length.out = 2*h, by = "day"))){
  cat(as.character(day), " ")
  tst <- te[date >= day - max_lags & date <= day]
  create_fea(tst)
  tst <- data.matrix(tst[date == day][, c("id", "sales", "date") := NULL])
  te[date == day, sales := predict(m_lgb, tst)]
}
toc()

#--- Save the predictions made as RDS
#saveRDS(te, file = "predictions_lgb_simpler.rds")
#te <- readRDS(file.path(path_data_clean, "predictions_lgb_simpler.rds"))

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

sqrt(mean((my_dataframe$actual - my_dataframe$pred)^2)) # RMSE = 2.14

rm(list=c("actual_val", "my_pred", "my_dataframe"))

########## GENERATING OUTPUT ##########

#--- Preparing output file
te[date >= fday #retain only the rows where the date is on or after "2016-04-25"
][date >= fday+h, id := sub("validation", "evaluation", id) #if the date is on or after fday + h (where h is 28), "validation" is substituted with "evaluation" in the id field.
][, d := paste0("F", 1:28), by = id #new column d is created with values "F1", ..., "F28" grouped by id.
][, dcast(.SD, id ~ d, value.var = "sales") #reshape the data table from a long format to a wide format
][, fwrite(.SD, "final_output.csv")] #write output file

#--- Housekeeping
rm(list=ls())
