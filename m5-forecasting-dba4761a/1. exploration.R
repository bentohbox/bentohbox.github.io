# Clearing the environment
#rm(list = ls())

# Setting directory
path_root <- ".."
path_data_raw <- file.path(path_root,"data/raw//")

# Importing datasets
prices <- fread(file.path(path_data_raw, "sell_prices.csv"))
cal <- fread(file.path(path_data_raw, "calendar.csv"))
sales <- fread(file.path(path_data_raw, "sales_train_validation.csv"), nrows = Inf)

# Checking for NAs in sales and price data
sum(is.na(sales)) # No NAs are present
sum(is.na(prices)) # No NAs are present

# Inspecting sales data by selecting first 10 rows and columns because file is too large
sales %>% select(c(1:10)) %>% 
  head(10) # %>% kable() %>% kable_styling() 
## We already see that there are many zero values

# Inspecting selling prices
prices %>% head(10) 
## store_id and item_id appear in the sales data as well

prices$sell_price %>% summary()
## items range from 10 cents to $107.32  

#--- EDA 1: Mean Sales by Store
sales_d <- select(sales, -c(state_id, store_id, cat_id, dept_id, item_id))

## Transpose 'sales_d' DataFrame
sales_p <- as.data.frame(t(sales_d[, -1]))
colnames(sales_p) <- sales_d$id

## Merge with 'cal' DataFrame based on the 'd' column
merged_data <- merge(x = data.frame(date = cal$d, row.names = NULL), 
                     y = data.frame(sales_p, row.names = rownames(sales_p)), 
                     by.x = "date", 
                     by.y = "row.names", 
                     all.y = TRUE)

rownames(merged_data) <- NULL

## Calculate mean sales for each store
stores <- unique(prices$store_id)
Means <- numeric(length(stores))

for (i in seq_along(stores)){
  store <- stores[i]
  store_items <- grep(store, colnames(merged_data), value = TRUE)
  Means[i] <- mean(rowSums(select(merged_data, one_of(store_items)), na.rm = TRUE))}

## Create a data frame for plotting
df <- data.frame("Mean_sales" = Means, "Stores" = stores)

## Plot
ggplot(df, aes(x = Stores, y = Mean_sales, fill = Stores)) +
  geom_bar(stat = "identity") + ggtitle("Stores vs Mean sales")

#--- EDA 2.1: Number of Items by Category
items_by_id <- sales %>%
  group_by(cat_id) %>%
  summarise(number_of_items = n()) %>%
  arrange(number_of_items)

# Create the bar chart
ggplot(items_by_id, aes(x = cat_id, y = number_of_items, fill = cat_id)) +
  geom_bar(stat = "identity") +
  ggtitle("Count of Items by Category") +
  xlab("Categories") +
  ylab("Number of Items") +
  theme_minimal()

#--- EDA 2.2: Sales of each Item by Category over Time
# Get unique categories
categories <- unique(sales$cat_id)

# Initialize an empty data frame to hold the summed sales
summed_sales <- data.frame(matrix(ncol = 0, nrow = nrow(sales_p)))

# Loop through each category and sum the sales
for(cat in categories) {
  cat_cols <- grep(cat, colnames(sales_p), value = TRUE)
  summed_sales[cat] <- rowSums(select(sales_p, one_of(cat_cols)), na.rm = TRUE)}

# Convert to long format for plotting
summed_sales_long <- gather(summed_sales, key = "Category", value = "Sales")

# Add date to the long format data frame
summed_sales_long$Date <- merged_data$date
summed_sales_long_1 <- summed_sales_long
colnames(summed_sales_long_1)[colnames(summed_sales_long_1) == "Date"] ="d"
summed_sales_long_1 <- summed_sales_long_1 %>% left_join(cal) %>% select(c("Category","Sales","d","date"))

# Time Series Plot
ggplot(summed_sales_long_1, aes(x = date, y = Sales, colour = Category, group = Category)) +
  geom_line() +
  ggtitle("Total Sales by Item Category") +
  xlab("Date") +
  ylab("Sales") +
  theme_minimal()

#--- EDA 3: Sales by Event Type by Category
# Convert sales data from wide to long format
sales_long <- sales %>%
  gather(key = "d", value = "sold", -c(id, item_id, dept_id, cat_id, store_id, state_id)) %>%
  drop_na()

# Joining sales_long, cal together  
df <- sales_long %>% left_join(cal, by = "d")    
# Combine data from event_type_1 and event_type_2 
event_data <- 
  bind_rows(df %>% 
              select(event_type_1, cat_id, sold) %>% 
              filter(!is.na(event_type_1) & event_type_1 != ""), df %>% select(event_type_2, cat_id, sold) %>% filter(!is.na(event_type_2) & event_type_2 != "") )  
# Calculate the total sales for each event type and category 
event_sales <- event_data %>%   group_by(event_type_1, cat_id) %>%   summarise(total_sales = sum(sold))  
# Create a bar graph to compare event types, ordering by total_sales 
ggplot(event_sales %>% filter(!is.na(event_type_1)), aes(x = reorder(event_type_1, -total_sales), y = total_sales, fill = cat_id)) +   geom_bar(stat = "identity", position = "dodge") +   ggtitle("Total Sales by Event Type") +   xlab("Event Type") +   ylab("Total Sales") +   theme_minimal()

#--- EDA 4: Sell Price Distribution (note: long plotting time)
# box plot for each state id
state_box_plot <- prices %>% mutate(state_id = sub("_.*", "", store_id)) %>% ggplot(aes(x = state_id, y = sell_price, color = state_id)) + geom_boxplot() + ggtitle("Selling Prices across States") + xlab("State IDs") + ylab("Selling Price") 
state_box_plot

# box plot for each store id
store_box_plot <- prices %>% ggplot(aes(x = store_id, y = sell_price, color = store_id)) + geom_boxplot() + ggtitle("Selling Prices across Stores") + xlab("Store IDs") + ylab("Selling Price") + theme(axis.text.x = element_text(angle = 45, hjust = 1))
store_box_plot

# box plot for each store id, specifically for each category
hobbies_box_plot <- prices %>% mutate(cat = sub("_.*", "",item_id)) %>% filter(cat == "HOBBIES") %>% ggplot(aes(x = store_id, y = sell_price, color = store_id)) + geom_boxplot() + ggtitle("Selling Prices for Hobbies Category across Stores") +   xlab("Store IDs") +   ylab("Selling Price") 
hobbies_box_plot  

food_box_plot <- prices %>% mutate(cat = sub("_.*", "",item_id)) %>% filter(cat == "FOODS") %>% ggplot(aes(x = store_id, y = sell_price, color = store_id)) + geom_boxplot() + ggtitle("Selling Prices for Food Category across Stores") +   xlab("Store IDs") +   ylab("Selling Price") 
food_box_plot  

house_box_plot <- prices %>% mutate(cat = sub("_.*", "",item_id)) %>% filter(cat == "HOUSEHOLD") %>% ggplot(aes(x = store_id, y = sell_price, color = store_id)) + geom_boxplot() + ggtitle("Selling Prices for Household Category across Stores") +   xlab("Store IDs") +   ylab("Selling Price") 
house_box_plot  

figure <- ggarrange(hobbies_box_plot, food_box_plot, house_box_plot) 
figure

#--- Housekeeping
rm(list=ls())