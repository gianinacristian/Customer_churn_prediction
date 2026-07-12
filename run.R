args <- commandArgs(trailingOnly = FALSE)
file_arg <- grep("^--file=", args, value = TRUE)
project_dir <- if (length(file_arg)) {
  dirname(normalizePath(sub("^--file=", "", file_arg[1])))
} else {
  getwd()
}

if (!requireNamespace("ranger", quietly = TRUE)) {
  stop("Run: Rscript install_packages.R")
}

set.seed(12345)
results_dir <- file.path(project_dir, "results")
dir.create(results_dir, showWarnings = FALSE)

train <- read.csv(file.path(project_dir, "data", "telco_train.csv"), check.names = FALSE)
test <- read.csv(file.path(project_dir, "data", "telco_test.csv"), check.names = FALSE)

prepare_data <- function(data) {
  data$CUSTOMER_TENURE_DAYS <- as.numeric(as.Date("2013-10-31") - as.Date(data$START_DATE))
  data$OUTCALLER <- as.integer(
    data$AVG_MINUTES_OUT_OFFNET_1MONTH > data$AVG_MINUTES_INC_OFFNET_1MONTH
  )

  zero_fill <- c("COUNT_CONNECTIONS_3MONTH", "AVG_DATA_1MONTH", "AVG_DATA_3MONTH")
  data[zero_fill] <- lapply(data[zero_fill], function(x) replace(x, is.na(x), 0))
  data[c("ID", "START_DATE", "FIN_STATE")] <- NULL
  data
}

fit_preprocessor <- function(data) {
  medians <- vapply(data, median, numeric(1), na.rm = TRUE)
  medians[!is.finite(medians)] <- 0

  filled <- data
  for (name in names(filled)) {
    filled[[name]][is.na(filled[[name]])] <- medians[[name]]
  }

  centers <- vapply(filled, mean, numeric(1))
  scales <- vapply(filled, sd, numeric(1))
  scales[!is.finite(scales) | scales == 0] <- 1

  list(medians = medians, centers = centers, scales = scales)
}

apply_preprocessor <- function(data, preprocessor) {
  data <- data[names(preprocessor$medians)]
  for (name in names(data)) {
    data[[name]][is.na(data[[name]])] <- preprocessor$medians[[name]]
    data[[name]] <- (data[[name]] - preprocessor$centers[[name]]) /
      preprocessor$scales[[name]]
  }
  data
}

auc <- function(actual, probability) {
  positives <- sum(actual == 1)
  negatives <- sum(actual == 0)
  ranks <- rank(probability, ties.method = "average")
  (sum(ranks[actual == 1]) - positives * (positives + 1) / 2) /
    (positives * negatives)
}

metrics <- function(actual, probability, model) {
  predicted <- as.integer(probability >= 0.5)
  tp <- sum(predicted == 1 & actual == 1)
  tn <- sum(predicted == 0 & actual == 0)
  fp <- sum(predicted == 1 & actual == 0)
  fn <- sum(predicted == 0 & actual == 1)
  precision <- if (tp + fp == 0) 0 else tp / (tp + fp)
  recall <- if (tp + fn == 0) 0 else tp / (tp + fn)
  specificity <- if (tn + fp == 0) 0 else tn / (tn + fp)

  data.frame(
    model = model,
    auc = auc(actual, probability),
    accuracy = (tp + tn) / length(actual),
    balanced_accuracy = (recall + specificity) / 2,
    precision = precision,
    recall = recall,
    f1 = if (precision + recall == 0) 0 else 2 * precision * recall / (precision + recall)
  )
}

train_ids <- train$ID
test_ids <- test$ID
y <- as.integer(train$CHURN)
x <- prepare_data(train[names(train) != "CHURN"])
test_x <- prepare_data(test)

validation_rows <- sort(unlist(lapply(split(seq_along(y), y), function(rows) {
  sample(rows, max(1, round(length(rows) * 0.2)))
})))
training_rows <- setdiff(seq_along(y), validation_rows)

preprocessor <- fit_preprocessor(x[training_rows, , drop = FALSE])
x_train <- apply_preprocessor(x[training_rows, , drop = FALSE], preprocessor)
x_validation <- apply_preprocessor(x[validation_rows, , drop = FALSE], preprocessor)
y_train <- y[training_rows]
y_validation <- y[validation_rows]

logistic_data <- data.frame(CHURN = y_train, x_train, check.names = FALSE)
logistic_model <- glm(CHURN ~ ., data = logistic_data, family = binomial())
logistic_probability <- as.numeric(
  predict(logistic_model, newdata = x_validation, type = "response")
)

class_counts <- table(factor(y_train, levels = c(0, 1)))
class_weights <- as.numeric(sum(class_counts) / (2 * class_counts))
names(class_weights) <- names(class_counts)
forest_data <- data.frame(CHURN = factor(y_train, levels = c(0, 1)), x_train, check.names = FALSE)
forest_model <- ranger::ranger(
  CHURN ~ .,
  data = forest_data,
  probability = TRUE,
  num.trees = 500,
  min.node.size = 5,
  class.weights = class_weights,
  importance = "impurity",
  seed = 12345
)
forest_probability <- as.numeric(
  predict(forest_model, data = x_validation)$predictions[, "1"]
)

model_metrics <- rbind(
  metrics(y_validation, logistic_probability, "logistic_regression"),
  metrics(y_validation, forest_probability, "random_forest")
)
model_metrics <- model_metrics[order(model_metrics$auc, decreasing = TRUE), ]
write.csv(model_metrics, file.path(results_dir, "model_metrics.csv"), row.names = FALSE)

best_model <- model_metrics$model[1]
best_probability <- if (best_model == "random_forest") forest_probability else logistic_probability
write.csv(
  data.frame(
    ID = train_ids[validation_rows],
    actual_churn = y_validation,
    logistic_probability = logistic_probability,
    random_forest_probability = forest_probability,
    selected_probability = best_probability
  ),
  file.path(results_dir, "validation_predictions.csv"),
  row.names = FALSE
)

ordered_actual <- y_validation[order(best_probability, decreasing = TRUE)]
roc <- data.frame(
  false_positive_rate = c(0, cumsum(ordered_actual == 0) / sum(ordered_actual == 0)),
  true_positive_rate = c(0, cumsum(ordered_actual == 1) / sum(ordered_actual == 1))
)
png(file.path(results_dir, "roc_curve.png"), width = 900, height = 650)
plot(
  roc$false_positive_rate,
  roc$true_positive_rate,
  type = "l",
  lwd = 2,
  xlab = "False positive rate",
  ylab = "True positive rate",
  main = paste("Validation ROC -", gsub("_", " ", best_model))
)
abline(0, 1, lty = 2)
dev.off()

full_preprocessor <- fit_preprocessor(x)
full_x <- apply_preprocessor(x, full_preprocessor)
full_test_x <- apply_preprocessor(test_x, full_preprocessor)

if (best_model == "random_forest") {
  full_counts <- table(factor(y, levels = c(0, 1)))
  full_weights <- as.numeric(sum(full_counts) / (2 * full_counts))
  names(full_weights) <- names(full_counts)
  final_data <- data.frame(CHURN = factor(y, levels = c(0, 1)), full_x, check.names = FALSE)
  final_model <- ranger::ranger(
    CHURN ~ .,
    data = final_data,
    probability = TRUE,
    num.trees = 500,
    min.node.size = 5,
    class.weights = full_weights,
    importance = "impurity",
    seed = 12345
  )
  test_probability <- as.numeric(predict(final_model, data = full_test_x)$predictions[, "1"])
  importance <- final_model$variable.importance
} else {
  final_data <- data.frame(CHURN = y, full_x, check.names = FALSE)
  final_model <- glm(CHURN ~ ., data = final_data, family = binomial())
  test_probability <- as.numeric(predict(final_model, newdata = full_test_x, type = "response"))
  importance <- abs(coef(final_model))
  importance <- importance[names(importance) != "(Intercept)"]
}

write.csv(
  data.frame(ID = test_ids, churn_probability = test_probability),
  file.path(results_dir, "test_predictions.csv"),
  row.names = FALSE
)

importance <- importance[is.finite(importance)]
feature_importance <- data.frame(
  feature = names(importance),
  importance = as.numeric(importance)
)
feature_importance <- feature_importance[order(feature_importance$importance, decreasing = TRUE), ]
write.csv(
  feature_importance,
  file.path(results_dir, "feature_importance.csv"),
  row.names = FALSE
)

cat("Selected model:", best_model, "\n")
print(model_metrics)
