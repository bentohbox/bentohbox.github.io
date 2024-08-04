#--- Preparing the dataset
training <- create_dt()
free()

#--- Run create_fea function after create_dt() based on which model you want to build, do not run multiple
tic() # 20 minutes to create features
create_fea_lgb_singular(training) # for lgb singular model
toc()

tic() #20 minutes to create features
create_fea_lgb_store(training) # for lgb dedicated store model
toc()

tic() #20 minutes to create features
create_fea(training) # for lgb with simple features
toc()

# Include macroeconomic factors: inflation, unemployment rate, gdp per capita
training <- merge(training, macroeconomic_data, by = c("year", "month", "quarter"), all.x = TRUE)
toc()
free()

training <- na.omit(training) # Removing NAs for model training
free()

#--- Save as RDS file after removing NAs. Change name accordingly.
#saveRDS(training, file = "training.rds")

#--- Read in "training.rds" for the training dataset singular
#training <- readRDS(file.path(path_data_clean, "training.rds"))

#--- Read in "training_store.rds" for the training dataset storewise
#training <- readRDS(file.path(path_data_clean, "training_store.rds"))

#--- Read in "training_lgb_simpler.rds" for the training dataset simpler features
#training <- readRDS(file.path(path_data_clean, "training_lgb_simpler.rds"))
