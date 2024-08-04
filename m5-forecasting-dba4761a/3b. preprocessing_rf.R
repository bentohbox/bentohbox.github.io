#--- Preparing the dataset
training <- create_dt()
free()

tic() # 20/30 minutes to create features
create_fea_rf(training)
toc()
# Include macroeconomic factors: inflation, unemployment rate, gdp per capita
training <- merge(training, macroeconomic_data, by = c("year", "month", "quarter"), all.x = TRUE)
free()

training[, date := as.Date(date)]

# Filter out NA values in 'sales' and 'date'
training <- training %>% 
  filter(!is.na(sales) & !is.na(date))

#--- Save as RDS file after removing NAs
#saveRDS(training, file = "training_rf.rds")
