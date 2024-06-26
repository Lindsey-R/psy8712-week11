# Script Settings and Resources
library(tidyverse)
library(haven) 
library(caret)
library(tictoc) 
library(parallel)
library(doParallel)
set.seed(12138)

# Data Import and Cleaning
GSS2016 <- read_sav(file = "GSS2016.sav") 
gss_tbl <- GSS2016 %>%
  filter(!is.na(MOSTHRS)) %>% 
  rename(`work hours` = MOSTHRS) %>% 
  select(-HRS1, -HRS2) %>% 
  select(where(function(x) (sum(is.na(x))/nrow(.)) < 0.75)) %>% 
  sapply(as.numeric)  %>%
  as_tibble()

# Analysis
Original <- rep(0,4)
Parallel <- rep(0,4)

random_sample <- sample(nrow(gss_tbl))
gss_shuffle_tbl <- gss_tbl[random_sample, ]
index <- round(nrow(gss_tbl) * 0.75)
gss_train_tbl <- gss_shuffle_tbl[1:index, ]
gss_test_tbl <- gss_shuffle_tbl[(index+1):nrow(gss_tbl), ]
fold_indices <- createFolds(gss_train_tbl$`work hours`, 10)
myControl <- trainControl(method = "cv", 
                          indexOut = fold_indices, 
                          number = 10,  
                          verboseIter = TRUE) 

tic()
model_ols <- train(`work hours` ~ ., 
                   gss_train_tbl,
                   method = "lm",  
                   preProcess = "medianImpute", 
                   na.action = na.pass, 
                   trControl = myControl)
ols_predict <- predict(model_ols, gss_test_tbl, na.action = na.pass)
time_ols = toc()
Original[1] = time_ols$toc - time_ols$tic # Calculate time


tic()
model_elastic <- train(`work hours` ~ ., 
                       data = gss_train_tbl,
                       method = "glmnet",  
                       preProcess = "medianImpute",
                       na.action = na.pass, 
                       trControl = myControl)
elastic_predict <- predict(model_elastic, gss_test_tbl, na.action = na.pass)
time_elastic = toc()
Original[2] = time_elastic$toc - time_elastic$tic # Calculate time


tic()
model_rf <- train(`work hours` ~ ., 
                  data = gss_train_tbl,
                  method = "ranger",  
                  preProcess = "medianImpute",
                  na.action = na.pass, 
                  trControl = myControl)
rf_predict <- predict(model_rf, gss_test_tbl, na.action = na.pass)
time_rf = toc()
Original[3] = time_rf$toc - time_rf$tic 

tic()
model_xgb <- train(`work hours` ~ ., 
                   data = gss_train_tbl,
                   method = "xgbLinear",  
                   preProcess = "medianImpute", 
                   na.action = na.pass, 
                   trControl = myControl)
xgb_predict <- predict(model_xgb, gss_test_tbl, na.action = na.pass)
time_xgb = toc()
Original[4] = time_xgb$toc - time_xgb$tic 


local_cluster <- makeCluster(30) # Modify to take 30 clusters - currently I have 7 
registerDoParallel(local_cluster)


tic()
model_ols_par <- train(`work hours` ~ ., 
                   gss_train_tbl,
                   method = "lm",  
                   preProcess = "medianImpute", 
                   na.action = na.pass, 
                   trControl = myControl)
ols_predict_par <- predict(model_ols_par, gss_test_tbl, na.action = na.pass)
time_ols_par = toc()
Parallel[1] = time_ols_par$toc - time_ols_par$tic # Calculate time


tic()
model_elastic_par <- train(`work hours` ~ ., 
                       data = gss_train_tbl,
                       method = "glmnet",  
                       preProcess = "medianImpute",
                       na.action = na.pass, 
                       trControl = myControl)
elastic_predict_par <- predict(model_elastic_par, gss_test_tbl, na.action = na.pass)
time_elastic_par = toc()
Parallel[2] = time_elastic_par$toc - time_elastic_par$tic 

tic()
model_rf_par <- train(`work hours` ~ ., 
                  data = gss_train_tbl,
                  method = "ranger",  
                  preProcess = "medianImpute",
                  na.action = na.pass, 
                  trControl = myControl)
rf_predict_par <- predict(model_rf_par, gss_test_tbl, na.action = na.pass)
time_rf_par = toc()
Parallel[3] = time_rf_par$toc - time_rf_par$tic 


tic()
model_xgb_par <- train(`work hours` ~ ., 
                   data = gss_train_tbl,
                   method = "xgbLinear",  
                   preProcess = "medianImpute", 
                   na.action = na.pass, 
                   trControl = myControl)
xgb_predict_par <- predict(model_xgb_par, gss_test_tbl, na.action = na.pass)
time_xgb_par = toc()
Parallel[4] = time_xgb_par$toc - time_xgb_par$tic


stopCluster(local_cluster)
registerDoSEQ


# Publication
R2_ols <- cor(ols_predict, gss_test_tbl$`work hours`)^2
R2_elastic <- cor(elastic_predict, gss_test_tbl$`work hours`)^2
R2_rf <- cor(rf_predict, gss_test_tbl$`work hours`) ^2
R2_xgb <- cor(xgb_predict, gss_test_tbl$`work hours`) ^2

`Table 3` <- tibble(
  algo = c("OLS regression", "Elastic Net", "Random Forest", "XGB"),
  cv_rsq = c(sub("^0\\.", ".", formatC(model_ols$results$Rsquared, format = 'f', digits = 2)),
             sub("^0\\.", ".", formatC(max(model_elastic$results$Rsquared), format = 'f', digits = 2)),
             sub("^0\\.", ".", formatC(max(model_rf$results$Rsquared), format = 'f', digits = 2)),
             sub("^0\\.", ".", formatC(max(model_xgb$results$Rsquared), format = 'f', digits = 2))),
  ho_rsq = c(sub("^0\\.", ".", formatC(R2_ols, format = 'f', digits = 2)),
             sub("^0\\.", ".", formatC(R2_elastic, format = 'f', digits = 2)),
             sub("^0\\.", ".", formatC(R2_rf, format = 'f', digits = 2)),
             sub("^0\\.", ".", formatC(R2_xgb, format = 'f', digits = 2)))
)

`Table 4` <- tibble(
  algo = c("OLS regression", "Elastic Net", "Random Forest", "XGB"),
  supercomputer = Original,
  'supercomputer_30' = Parallel
)

# Save as csv
write_csv(`Table 3`, "table3.csv")
write_csv(`Table 4`, "table4.csv")

## Answer questions
# 1. It cut the most for random forest (also a lot for XGB). Random forest takes 
# the longest time on my local computer,  I think it is due to more cores make
# it easier to build trees and calculate results.

# 2. More cores, less time.

# 3. I will definitely do. In my condition I wait 4 minutes for the supercomputer 
# to process the result and give it back - it is much faster than my personal laptop,
# and if I'm testing multiple models at once it will work even better and more
# time efficient. However, For easy models, such as only fitting lm and glmnet, I think 
# there is no need for supercomputer. 
