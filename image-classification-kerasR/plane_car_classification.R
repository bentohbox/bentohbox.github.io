library(EBImage)
library(keras3)
library(tensorflow)

# Import to_categorical correctly from tensorflow.keras.utils
to_categorical <- tensorflow::tf$keras$utils$to_categorical

# Set working directory
setwd('C:\\Users\\bened\\OneDrive\\Data Science\\image-classification-kerasR')
path_root <-  "."
path_planes <- file.path(path_root, "planes")
path_cars <- file.path(path_root, "cars")

# Function to read images from a given directory and return a list of images
read_images <- function(path, pattern = "\\.jpg$") {
  files <- list.files(path, pattern = pattern, full.names = TRUE)
  pics <- list()
  for (i in 1:length(files)) {
    tryCatch({
      pics[[i]] <- readImage(files[i])
    }, error = function(e) {
      message("Error reading file ", files[i], ": ", e)
    })
  }
  return(pics)
}

# Read plane and car images
planes_pic <- read_images(path_planes)
cars_pic <- read_images(path_cars)

# Combine the lists
mypic <- c(planes_pic, cars_pic)

# Filter out NULL values
mypic <- Filter(Negate(is.null), mypic)

# Convert images to grayscale
for (i in 1:length(mypic)) {
  mypic[[i]] <- channel(mypic[[i]], "gray")
}

# Exploring
print(mypic[[1]])
display(mypic[[500]])
summary(mypic[[3]])
hist(mypic[[11]])
str(mypic[[15]])

# Resize my images to same size - 64x64
for (i in 1:length(mypic)) {
  mypic[[i]] <- resize(mypic[[i]], 64, 64)
}

# Convert to matrices by reshaping - obtain a vector for each image
for (i in 1:length(mypic)) {
  mypic[[i]] <- array_reshape(mypic[[i]], c(64, 64, 1))
}

# Creating our dataset using rbind
trainx <- NULL
# Designate last 10 for test for both vehicles
# planes - 1 to 315/10, 316 to 325
for (i in 1:315) {
  trainx <- rbind(trainx, mypic[[i]])
}

# cars - 326 to 731/10, 732 to 741
for (i in 326:731) {
  trainx <- rbind(trainx, mypic[[i]])
}

str(trainx)

testx <- rbind(do.call(rbind, mypic[316:325]), do.call(rbind, mypic[732:741]))

str(testx)

# Create labels: 0 - plane, 1 - car
trainy <- as.integer(c(rep(0, times = 315), rep(1, times = 406)))
testy <- as.integer(c(rep(0, times = 10), rep(1, times = 10)))
num_classes <- 2  # Assuming binary classification

# One-hot encoding
trainLabels <- to_categorical(as.integer(trainy), num_classes = as.integer(num_classes))
testLabels <- to_categorical(as.integer(testy), num_classes = as.integer(num_classes))

model <- keras_model_sequential()
model %>%
  layer_dense(units = 256, activation = 'relu', input_shape = c(4096)) %>%
  layer_dense(units = 128, activation = 'relu') %>%
  layer_dense(units = 64, activation = 'relu') %>%
  layer_dense(units = 2, activation = 'softmax')

# Compile the model using the correct function
model %>% compile(
  loss = 'binary_crossentropy',
  optimizer = optimizer_rmsprop(),
  metrics = c('accuracy')
)

# fit the model
history <- model %>%
  fit(trainx,
      trainLabels,
      epochs = 30,
      batch_size = 32,
      validation_split = 0.2)

plot(history)

# evaluation on train data
model %>% evaluate(trainx, trainLabels)
pred <- model %>% predict(trainx)
predicted_classes <- apply(pred, 1, which.max) - 1
table(predicted = predicted_classes, actual = trainy)

# evaluation on test data
model %>% evaluate(testx, testLabels)
pred <- model %>% predict(testx)
predicted_classes <- apply(pred, 1, which.max) -1
table(predicted = predicted_classes, actual = testy)
