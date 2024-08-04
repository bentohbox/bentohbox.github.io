#--- Setting up data path
path_root <- ".."
path_data_raw <- file.path(path_root,"data/raw//")
path_data_clean <- file.path(path_root,"data/clean//")
path_model <- file.path(path_root,"model//")

#--- Seed for Reproducibility 
set.seed(4761)  

# Alpha parameter for exponential smoothing
alpha <- 0.9

#--- Loading data
te_naive <- create_dt(FALSE)  

############### Without exponential smoothing ###############

# Measure the time taken by the basic naive model
time_naive <- system.time({   
  
  # Loop through each day for prediction   
  for (day in as.list(seq(fday, length.out = 2 * h, by = "day"))) {     
    cat(as.character(day), " ")          
    
    # Use the last known value for each time series as the prediction     
    te_naive[date == day, sales := te_naive[date == day - 1, sales]]   } })[3]  


# Calculate RMSE for the naive model 
my_pred_naive <- te_naive[date >= fday & date < fday + 28] %>% 
  select(sales) 

actual_val_naive <- fread(file.path(path_data_raw, "sales_train_evaluation.csv"), nrows = Inf) %>%   
  select(-c(7:1919)) %>%   
  melt(measure.vars = patterns("^d_"), variable.name = "d", value.name = "sales") %>%   
  select(sales)  

my_dataframe_naive <- data.table(my_pred_naive, actual_val_naive) 
colnames(my_dataframe_naive) <- c("pred", "actual")  

# Calculate RMSE without exponential smoothing
rmse_naive <- sqrt(mean((my_dataframe_naive$actual - my_dataframe_naive$pred)^2))


############### With exponential smoothing ###############

# Measure the time taken by the naive model with simple exponential smoothing
time_naive_with_smoothing <- system.time({   
  # Loop through each day for prediction
  for (day in as.list(seq(fday, length.out = 2 * h, by = "day"))) {     
    cat(as.character(day), " ")          
    
    # Get the last known value
    last_known_value <- ifelse(day == fday, 
                               te_naive[date == (day - 1), sales], 
                               te_naive[date == day, exp_smoothed_sales])
    
    # Exponential smoothing
    te_naive[date == day, exp_smoothed_sales := (alpha * last_known_value) + ((1 - alpha) * last_known_value)]
  } 
})[3]

# Calculate RMSE with exponentially smoothed sales
my_pred_smoothed <- te_naive[date >= fday & date < fday + 28, .(exp_smoothed_sales)]
actual_val_naive <- fread(file.path(path_data_raw, "sales_train_evaluation.csv"), nrows = Inf) %>%   
  select(-c(7:1919)) %>%   
  melt(measure.vars = patterns("^d_"), variable.name = "d", value.name = "sales")

# Create a d vs date reference Dataframe, mapping each d to date 
d_date_ref <- actual_val_naive %>% 
  distinct(d, .keep_all = TRUE) %>% 
  select("d") %>% 
  mutate(date = fday + 0:(nrow(.) - 1))

# Adding date col to actual value naive Dataframe
actual_val_naive <- actual_val_naive %>% 
  left_join(d_date_ref, by = "d")

# Extracting actual values
actual_val_naive <- actual_val_naive[date >= fday & date < fday + 28, .(sales)]

# Calculate RMSE with exponential smoothing
rmse_naive_with_smoothing <- sqrt(mean((actual_val_naive$sales - my_pred_smoothed$exp_smoothed_sales)^2, na.rm = TRUE))

# Results without smoothing
cat("Naive Model RMSE:", rmse_naive, "\n") 
#Naive model w/o smoothing:RMSE = 2.893557
cat("Naive Model Time Elapsed:", time_naive, "seconds\n")

# Results with smoothing
cat("Naive Model with Exponential Smoothing RMSE:", rmse_naive_with_smoothing, "\n") 
#Naive model with smoothing:RMSE = 3.248972
cat("Naive Model with Exponential Smoothing Time Elapsed:", time_naive_with_smoothing, "seconds\n")

#---saving the predictions
#saveRDS(te_naive, file = "predictions_naive.rds")

#saveRDS(list(my_pred_naive, my_pred_smoothed), file = "m_naive_final.rds")

#--- Preparing output file
te_naive[date >= fday #retain only the rows where the date is on or after "2016-04-25"
][date >= fday+h, id := sub("validation", "evaluation", id) #if the date is on or after fday + h (where h is 28), "validation" is substituted with "evaluation" in the id field.
][, d := paste0("F", 1:28), by = id #new column d is created with values "F1", ..., "F28" grouped by id.
][, dcast(.SD, id ~ d, value.var = "sales") #reshape the data table from a long format to a wide format
][, fwrite(.SD, "final_output_naive.csv")] #write output file

#--- Finally, clear the environment
rm(list=ls())
