library(EBImage)
library(keras)


#Read in photos

photos <- c('rose1.jpg', 'rose2.jpg', 'rose3.jpg', 'rose4.jpg', 'rose5.jpg',
            'rose6.jpg', 'rose7.jpg', 'rose8.jpg', 'rose9.jpg', 'rose10.jpg',
            'daisy1.jpg', 'daisy2.jpg', 'daisy3.jpg', 'daisy4.jpg', 'daisy5.jpg',
            'daisy6.jpg', 'daisy7.jpg', 'daisy8.jpg', 'daisy9.jpg', 'daisy10.jpg',
            'pansy1.jpg', 'pansy2.jpg', 'pansy3.jpg', 'pansy4.jpg', 'pansy5.jpg',
            'pansy6.jpg', 'pansy7.jpg', 'pansy8.jpg', 'pansy9.jpg', 'pansy10.jpg') 

my_photos <- list()

photos_size <- length(photos)

for(i in 1:photos_size) {my_photos[[i]] <- readImage(photos[i])}

#Resize

for(i in 1:photos_size) {my_photos[[i]] <- resize(my_photos[[i]], 28, 28)}

#Reshape

for(i in 1:photos_size) {my_photos[[i]] <- array_reshape(my_photos[[i]], c(28, 28, 3))}

str(my_photos)

#Row Bind

trainx <- NULL

for(i in 1:9) {trainx <- rbind(trainx, my_photos[[i]])}

for(i in 11:19) {trainx <- rbind(trainx, my_photos[[i]])}

for(i in 21:29) {trainx <- rbind(trainx, my_photos[[i]])}

testx <- rbind(my_photos[[10]], my_photos[[20]], my_photos[[30]])

trainy <- c(0, 0, 0, 0, 0, 0, 0, 0, 0,
            1, 1, 1, 1, 1, 1, 1, 1, 1,
            2, 2, 2, 2, 2, 2, 2, 2, 2)

testy <- c(0, 1, 2) 

#Encoding
 
trainLabels <- to_categorical(trainy)

testLabels <- to_categorical(testy)


#Model
model <- keras_model_sequential()

model %>% 
  layer_dense(units = 256, activation = 'relu', input_shape = c(2352)) %>% 
  layer_dense(units = 128, activation = 'relu') %>% 
  layer_dense(units = 3, activation = 'softmax')

summary(model)

#Compile
model %>% 
  compile(loss = 'categorical_crossentropy',
          optimizer = optimizer_rmsprop(),
          metrics = c('accuracy'))

#Fit Model
history <- model %>% 
  fit(trainx,
      trainLabels,
      epochs = 30,
      batch_size = 32,
      validation_split = 0.2)

plot(history)


#Evaluation & Prediction - train data
model %>% evaluate(trainx, trainLabels)

pred <- model %>% predict_classes(trainx)
table(Predicted = pred, Actual = trainy)

prob <- model %>% predict_proba(trainx)

cbind(prob, Predicted = pred, Actual = trainy)

#Evaluation & Prediction - test data

model %>% evaluate(testx, testLabels)

pred <- model %>%  predict_classes(testx)
table(Predicted = pred, Actual = testy)
