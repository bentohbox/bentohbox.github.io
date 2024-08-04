#--- Read in "training.rds" for the training dataset storewise
training <- readRDS(file.path(path_data_clean, "training_store.rds"))

#subset with only 2013 onwards
sub_date <- as.IDate("2013-01-01")
tr_subset <- training %>% filter(date>= sub_date)

#store_id: CA_1 to CA_4 = 1 to 4, TX_1 to TX_3 = 5 to 7, WI_1 to WI_3 = 8 to 10
stores <- c(1:10)

#split dataset into individual store data
list_of_dfs <- split(tr_subset, tr_subset$store_id)

#initialising a counter
counter <- 1

#iterating over the list of dataframes
for (sub_df in list_of_dfs) {
  assign(paste0("df", counter), sub_df)
  counter <- counter + 1
  if (counter > 10) {
    break
  }}

rm("tr_subset")
free()

list_of_dfs <- list(df1, df2, df3, df4, df5, df6, df7, df8, df9, df10)

categorical_v <- c("item_id", "state_id", "dept_id", "cat_id",
                   "wday", "mday", "week", "month", "quarter", "year", "is_weekend",
                   "snap_CA", "snap_TX", "snap_WI")

########## MODEL TRAINING ##########
set.seed(4761)

params <- list(
  boosting_type = "gbdt",
  objective = "tweedie", #sales data contains many zero counts, which can be modeled well with the Tweedie distribution 
  tweedie_variance_power = 1.1,
  metric = "rmse", #we want to minimise rmse as our objective
  subsample = 0.5, #subsampling helps to reduce training time without substantially compromising the learning process
  subsample_freq = 1, #prevent overfitting since the model sees a different subset of data at each step
  learning_rate = 0.03, #small enough to be able to achieve optimality without, large enough to reduce computing time
  num_leaves = 2047,
  min_data_in_leaf = 4095, # large minimum data in leaf helps in controlling overfitting
  feature_fraction = 0.5,
  max_bin = 100,
  n_estimators = 800,
  boost_from_average = FALSE #target variable has many zeros, prevents the model from being biased towards the mean sales
)

models <- list()

tic() # 1.5 hours
for (i in 1:length(list_of_dfs)) {
  training <- list_of_dfs[[i]]
  
  if (any(class(training$date) == "IDate")) {
    training$date <- as.integer(training$date)
  }
  
  # Extracting sales numbers for model training as the dependent variable
  y <- training$sales 
  
  # Obtaining the indices of the training set minus the most recent 28 days
  x_id <- training[date <= max(date) - 28, which = TRUE]
  
  # Removing id, sales, date
  training[, c("id", "sales", "date") := NULL]
  free()
  
  # Converting to a matrix for training
  training <- data.matrix(training)
  free()
  
  # Creating the LightGBM datasets with categorical features
  x_train <- lgb.Dataset(training[x_id, ], label = y[x_id], categorical_feature = categorical_v) 
  x_val <- lgb.Dataset(training[-x_id, ], label = y[-x_id], categorical_feature = categorical_v)
  
  # Train the model
  model <- lgb.train(params = params,
                     data = x_train,
                     nrounds = 800,
                     valids = list(train = x_train, valid = x_val),
                     early_stopping_rounds = 300,
                     eval_freq = 100)
  
  # Store the model
  models[[i]] <- model
  
  free()
}
toc()

#---feature importance
# we use lapply to apply lgb.importance() to each of the 10 models 
feature_importances <- lapply(models, function(model) {
  imp <- lgb.importance(model)
  data.frame(Feature = imp$Feature, Importance = imp$Gain)})

# bind all data frames into one and then aggregate by feature
# essentially, we are averaging the feature importances across all the 10 models
all_importances <- do.call(rbind, feature_importances)
aggregated_importances <- aggregate(Importance ~ Feature, all_importances, mean)

# plot
ggplot(aggregated_importances, aes(x=reorder(Feature, Importance), y=Importance)) +
  geom_bar(stat="identity") +
  coord_flip() +
  theme_minimal() +
  xlab("Feature") +
  ylab("Average Importance")

#---save models
#for (i in 1:length(models)) {
 # model <- models[[i]]
  #file_name <- paste0("model_", i, ".rds")  # Naming each model file
  #saveRDS.lgb.Booster(model, file = file_name)
#}

#---load the 10 models for each store
models <- list()
for (i in 1:10) {
  model_path <- paste0(path_model,"model_", i, ".rds")
  if (file.exists(model_path)) {
    models[[i]] <- readRDS.lgb.Booster(model_path)
  } else {
    cat("Model file", model_path, "not found\n")
  }
}

free()

########## MAKING PREDICTIONS ##########

#--- Creating our forecasts for LGB Dedicated Stores
tic() #2hours
te <- create_dt(FALSE) #preparing test set

macroeconomic_data <- fread(file.path(path_data_raw, "MacroeconomicFeatures.csv"))
macroeconomic_data$year <- as.integer(macroeconomic_data$year)
macroeconomic_data$month <- as.integer(macroeconomic_data$month)
macroeconomic_data$quarter <- as.integer(macroeconomic_data$quarter)

te$store_id <- as.integer(factor(te$store_id))
te$store_id <- as.integer(te$store_id)
free()

for (day in as.list(seq(fday, length.out = 2*h, by = "day"))) {
  cat(as.character(day), " ")
  tst <- te[date >= day - max_lags & date <= day]
  create_fea_lgb_store(tst)
  tst <- merge(tst, macroeconomic_data, by = c("year", "month", "quarter"), all.x = TRUE)
  tst_matrix <- data.matrix(tst[date == day][, c("id", "sales", "date") := NULL])
  
  # find the column index for 'store_id'
  store_id_col_index <- which(colnames(tst_matrix) == "store_id")
  
  # predict for each store using their respective models
  for (i in 1:10) {
    #filter test data for the current store_id within tst_matrix
    tst_store_matrix <- tst_matrix[tst_matrix[, store_id_col_index] == i, ]
    
    if (nrow(tst_store_matrix) > 0) {
      #predict using the store model
      te[date == day & te$store_id == i, sales := predict(models[[i]], tst_store_matrix)]
    }}}
toc()

#--- saving the predictions
#saveRDS(te, file = "predictions_lgb_store.rds")
free()

#--- Load in the predictions made for LGB Dedicated Store
#te <- readRDS(file.path(path_data_clean,"predictions_lgb_store.rds"))

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

sqrt(mean((my_dataframe$actual - my_dataframe$pred)^2)) #2.31

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
