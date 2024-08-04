#--- Setting up data path
path_root <- ".."
path_data_raw <- file.path(path_root,"data/raw//")
path_data_clean <- file.path(path_root,"data/clean//")
path_model <- file.path(path_root,"model//")

#--- Parameters
h <- 28 # Number of days to forecast (Horizon)
max_lags <- 366 
tr_last <- 1913 # Last day, d_1913
fday <- as.IDate("2016-04-25") # Date to forecast 28 days from

#--- Defining our functions for data preprocessing
free <- function() invisible(gc()) # This function allows us to free up unused memory

# create_dt reads the datasets and merges them into one dataframe for model training
create_dt <- function(is_train = TRUE, nrows = Inf) { 
  
  prices <- fread(file.path(path_data_raw, "sell_prices.csv")) # fread is used because its fast
  cal <- fread(file.path(path_data_raw, "calendar.csv"))
  cal[, `:=`(date = as.IDate(date, format="%Y-%m-%d"),
             is_weekend = as.integer(weekday %chin% c("Saturday", "Sunday")))] 
  # %chin% works like base R’s %in% but is optimized for speed and is for character vectors only
  
  if (is_train) {
    dt <- fread(file.path(path_data_raw, "sales_train_validation.csv"), nrows = nrows)
  } else {
    dt <- fread(file.path(path_data_raw, "sales_train_validation.csv"), nrows = Inf,
                drop = paste0("d_", 1:(tr_last-max_lags)))
    dt[, paste0("d_", (tr_last+1):(tr_last+2*h)) := NA_real_]}
  
  dt <- melt(dt, measure.vars = patterns("^d_"), # use regex to identify columns to melt
             variable.name = "d", value.name = "sales") # convert sales data from wide to long format
  
  # merging with calendar data
  dt <- dt[cal, `:=`(date = i.date, is_weekend = i.is_weekend,
                     wm_yr_wk = i.wm_yr_wk, event_name_1 = i.event_name_1,
                     snap_CA = i.snap_CA, snap_TX = i.snap_TX,
                     snap_WI = i.snap_WI), on = "d"]
  
  # merging with selling price data
  dt[prices, sell_price := i.sell_price, on = c("store_id", "item_id", "wm_yr_wk")]}

# Loading in macroeconomic data (inflation, unemployment, gdp per capita)
macroeconomic_data <- fread(file.path(path_data_raw, "MacroeconomicFeatures.csv"))
macroeconomic_data$year <- as.integer(macroeconomic_data$year)
macroeconomic_data$month <- as.integer(macroeconomic_data$month)
macroeconomic_data$quarter <- as.integer(macroeconomic_data$quarter)

# This function contains the features that we intend to include into our model
create_fea <- function(dt) { 
  
  # Creating 3 lag features
  lag <- c(7, 14, 28)
  dt[, (paste0("lag_", lag)) := data.table::shift(.SD, lag), .SDcols = "sales", by = "id"]
  
  # Creating rolling means using values from lag_28
  win <- c(7, 30, 90, 180)
  dt[, (paste0("roll_mean_28_", win)) := frollmean(lag_28, win, na.rm = TRUE), by = "id"]
  
  # Creating rolling max and variance with a 28 day window
  win <- c(28)
  dt[, (paste0("roll_max_28_", win)) := frollapply(lag_28, win, max), by = "id"]
  dt[, (paste0("roll_var_28_", win)) := frollapply(lag_28, win, var), by = "id"]
  
  # Creating a feature capturing changes in price
  dt[, price_change_1 := sell_price / data.table::shift(sell_price) - 1, by = "id"]
  dt[, price_change_365 := sell_price / frollapply(data.table::shift(sell_price), 365, max) - 1, by = "id"]
  
  # Converting categorical features into integers for model training
  cols <- c("item_id", "state_id", "dept_id", "cat_id", "event_name_1")   
  dt[, (cols) := lapply(.SD, function(x) as.integer(factor(x))), .SDcols = cols]
  
  dt[, `:=`(wday = wday(date), mday = mday(date),
            week = week(date), month = month(date),
            quarter = quarter(date), year = year(date),
            store_id = NULL, d = NULL,
            wm_yr_wk = NULL)]}

# This function contains the features that we intend to include into our model for LGB Singular
create_fea_lgb_singular <- function(dt) { 
  
  # Creating lag features (Sales)
  lag <- c(7, 14, 28)
  dt[, (paste0("lag_", lag)) := data.table::shift(.SD, lag), .SDcols = "sales", by = "id"]
  
  # Creating rolling means using values from lag_28 (Sales)
  win <- c(7, 30, 90, 180)
  dt[, (paste0("roll_mean_28_", win)) := frollmean(lag_28, win, na.rm = TRUE), by = "id"]
  
  # Creating rolling max and variance with a 28 day window (Sales)
  win <- c(28)
  dt[, (paste0("roll_max_28_", win)) := frollapply(lag_28, win, max), by = "id"]
  dt[, (paste0("roll_var_28_", win)) := frollapply(lag_28, win, var), by = "id"]
  
  # Creating a feature capturing changes in price (Price)
  dt[, price_change_1 := sell_price / data.table::shift(sell_price) - 1, by = "id"]
  dt[, price_change_365 := sell_price / frollapply(data.table::shift(sell_price), 365, max) - 1, by = "id"]
  
  # Converting categorical features into integers for model training
  cols <- c("item_id", "state_id", "dept_id", "cat_id", "event_name_1")   
  dt[, (cols) := lapply(.SD, function(x) as.integer(factor(x))), .SDcols = cols]
  
  # Extracts the weekday, day of month, week of the year, month, quarter, and year from the date (Calendar)
  dt[, `:=`(wday = wday(date), mday = mday(date),
            week = week(date), month = month(date),
            quarter = quarter(date), year = year(date),
            store_id = NULL, 
            d = NULL,
            wm_yr_wk = NULL)]
  
  # Counts number of days with zero sales and resets once there is positive sale
  dt[, continuous_zero_sales := 0]
  dt[, continuous_zero_sales := {
    zero_count = 0
    sapply(sales, function(s) {
      if (!is.na(s) && s == 0) {
        zero_count <<- zero_count + 1
      } else {
        zero_count <<- 0
      }
      zero_count
    })
  }, by = item_id]
}

# This function contains the features that we intend to include into our model for LGB Dedicated Store
create_fea_lgb_store <- function(dt) { 
  
  # Creating lag features (Sales)
  lag <- c(7, 14, 28)
  dt[, (paste0("lag_", lag)) := data.table::shift(.SD, lag), .SDcols = "sales", by = "id"]
  
  # Creating rolling means using values from lag_28 (Sales)
  win <- c(7, 30, 90, 180)
  dt[, (paste0("roll_mean_28_", win)) := frollmean(lag_28, win, na.rm = TRUE), by = "id"]
  
  # Creating rolling max and variance with a 28 day window (Sales)
  win <- c(28)
  dt[, (paste0("roll_max_28_", win)) := frollapply(lag_28, win, max), by = "id"]
  dt[, (paste0("roll_var_28_", win)) := frollapply(lag_28, win, var), by = "id"]
  
  # Creating a feature capturing changes in price (Price)
  dt[, price_change_1 := sell_price / data.table::shift(sell_price) - 1, by = "id"]
  dt[, price_change_365 := sell_price / frollapply(data.table::shift(sell_price), 365, max) - 1, by = "id"]
  
  # Converting categorical features into integers for model training
  cols <- c("item_id", "state_id", "dept_id", "cat_id", "event_name_1", "store_id")   
  dt[, (cols) := lapply(.SD, function(x) as.integer(factor(x))), .SDcols = cols]
  
  # Extracts the weekday, day of month, week of the year, month, quarter, and year from the date (Calendar)
  dt[, `:=`(wday = wday(date), mday = mday(date),
            week = week(date), month = month(date),
            quarter = quarter(date), year = year(date),
            d = NULL,
            wm_yr_wk = NULL)]
  
  # Counts number of days with zero sales and resets once there is positive sale
  dt[, continuous_zero_sales := 0]
  dt[, continuous_zero_sales := {
    zero_count = 0
    sapply(sales, function(s) {
      if (!is.na(s) && s == 0) {
        zero_count <<- zero_count + 1
      } else {
        zero_count <<- 0
      }
      zero_count
    })
  }, by = item_id]
}

# This function contains the features that we intend to include into our model for RF
create_fea_rf <- function(dt) { 
  
  # Creating lag features (Sales)
  lag <- c(1, 7, 14, 28)
  dt[, (paste0("lag_", lag)) := data.table::shift(.SD, lag), .SDcols = "sales", by = "id"]
  
  # Creating rolling means using values from lag_28 (Sales)
  win <- c(7, 30, 90, 180)
  dt[, (paste0("roll_mean_28_", win)) := frollmean(lag_28, win, na.rm = TRUE), by = "id"]
  
  # Creating rolling max and variance with a 28 day window (Sales)
  win <- c(28)
  dt[, (paste0("roll_max_28_", win)) := frollapply(lag_28, win, max), by = "id"]
  dt[, (paste0("roll_var_28_", win)) := frollapply(lag_28, win, var), by = "id"]
  
  # Creating a feature capturing changes in price (Price)
  dt[, price_change_1 := sell_price / data.table::shift(sell_price) - 1, by = "id"]
  dt[, price_change_365 := sell_price / frollapply(data.table::shift(sell_price), 365, max) - 1, by = "id"]
  
  # Converting categorical features into integers for model training
  cols <- c("item_id", "state_id", "dept_id", "cat_id", "event_name_1", "store_id")   
  dt[, (cols) := lapply(.SD, function(x) as.integer(factor(x))), .SDcols = cols]
  
  # Extracts the weekday, day of month, week of the year, month, quarter, and year from the date (Calendar)
  dt[, `:=`(wday = wday(date), mday = mday(date),
            week = week(date), month = month(date),
            quarter = quarter(date), year = year(date),
            #store_id = NULL, 
            d = NULL,
            wm_yr_wk = NULL)]
  
  # Counts number of days with zero sales and resets once there is positive sale
  dt[, continuous_zero_sales := 0]
  dt[, continuous_zero_sales := {
    zero_count = 0
    sapply(sales, function(s) {
      if (!is.na(s) && s == 0) {
        zero_count <<- zero_count + 1
      } else {
        zero_count <<- 0
      }
      zero_count
    })
  }, by = item_id]
}
