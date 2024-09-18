library(data.table)
library(arrow)



# UTILS -------------------------------------------------------------------
# Snake to camel for lighgbm
snakeToCamel <- function(snake_str) {
  # Replace underscores with spaces
  spaced_str <- gsub("_", " ", snake_str)

  # Convert to title case using tools::toTitleCase
  title_case_str <- tools::toTitleCase(spaced_str)

  # Remove spaces and make the first character lowercase
  camel_case_str <- gsub(" ", "", title_case_str)
  camel_case_str <- sub("^.", tolower(substr(camel_case_str, 1, 1)), camel_case_str)

  # I haeve added this to remove dot
  camel_case_str <- gsub("\\.", "", camel_case_str)

  return(camel_case_str)
}


# IMPORT DATA -------------------------------------------------------------
# Import raw local data
f = "/home/sn/data/equity/us/predictors_daily/ohlcv_predictors_sample_by_symbols.parquet"
dt = read_parquet(f)

# TODO: Remove some new data we dont have predictors for
dt = dt[date < as.Date("2024-07-01")]

# Inspect
dim(dt)


# LABELING ----------------------------------------------------------------
# Target variables
setorder(dt, symbol, date)
dt[, target_ret_1 := shift(close, 1, type = "lead") / close - 1, by = symbol]
dt[, .(symbol, date, close, target_ret_1)]
target_columns = c("target_ret_1")
dt = na.omit(dt, cols = target_columns)


# FEATURES SPACE ----------------------------------------------------------
# Features space from features raw
cols_non_features = c(
  "symbol", "date", "open", "high", "low", "close", "volume", "close_raw",
  "returns", target_columns)
cols_non_features %in% colnames(dt)
cols_features = setdiff(colnames(dt), cols_non_features)
cols = c(cols_non_features, cols_features)

# change feature and targets columns names due to lighgbm
cols_features_new = vapply(cols_features, snakeToCamel, FUN.VALUE = character(1L), USE.NAMES = FALSE)
setnames(dt, cols_features, cols_features_new)
cols_features = cols_features_new
cols_targets_new = vapply(target_columns, snakeToCamel, FUN.VALUE = character(1L), USE.NAMES = FALSE)
setnames(dt, target_columns, cols_targets_new)
target_columns = target_columns


# CLEAN DATA --------------------------------------------------------------
# Convert columns to numeric. This is important only if we import existing features
chr_to_num_cols = setdiff(colnames(dt[, .SD, .SDcols = is.character]), cols_non_features)
if (length(chr_to_num_cols) != 0) {
  dt = dt[, (chr_to_num_cols) := lapply(.SD, as.numeric), .SDcols = chr_to_num_cols]
}
log_to_num_cols = setdiff(colnames(dt[, .SD, .SDcols = is.logical]), cols_non_features)
if (length(log_to_num_cols) != 0) {
  dt = dt[, (log_to_num_cols) := lapply(.SD, as.numeric), .SDcols = log_to_num_cols]
}

# Remove duplicates
any_duplicates = any(duplicated(dt[, .(symbol, date)]))
if (any_duplicates) dt = unique(dt, by = c("symbol", "date"))

# Remove columns with many NA
keep_cols = names(which(colMeans(!is.na(dt)) > 0.80))
print(paste0("Removing columns with many NA values: ",
             setdiff(colnames(dt), c(keep_cols, "right_time"))))
dt = dt[, .SD, .SDcols = keep_cols]

# Remove Inf and Nan values if they exists
is.infinite.data.frame = function(x) do.call(cbind, lapply(x, is.infinite))
keep_cols = names(which(colMeans(!is.infinite(as.data.frame(dt))) > 0.95))
print(paste0("Removing columns with Inf values: ", setdiff(colnames(dt), keep_cols)))
dt = dt[, .SD, .SDcols = keep_cols]

# Remove inf values
n_0 = nrow(dt)
dt = dt[is.finite(rowSums(dt[, .SD, .SDcols = is.numeric], na.rm = TRUE))]
n_1 = nrow(dt)
print(paste0("Removing ", n_0 - n_1, " rows because of Inf values"))

# Remove constant columns in set
features_ = dt[, .SD, .SDcols = intersect(cols_features, colnames(dt))]
remove_cols = colnames(features_)[apply(features_, 2, var, na.rm=TRUE) == 0]
print(paste0("Removing constant: ", remove_cols))
dt = dt[, .SD, .SDcols = setdiff(colnames(dt), remove_cols)]

# Convert variables with low number of unique values to factors
cols_features = setdiff(colnames(dt), cols_non_features)
int_numbers = na.omit(dt[, ..cols_features])[, lapply(.SD, function(x) all(floor(x) == x))]
int_cols = colnames(dt[, ..cols_features])[as.matrix(int_numbers)[1,]]
factor_cols = dt[, ..int_cols][, lapply(.SD, function(x) length(unique(x)))]
factor_cols = as.matrix(factor_cols)[1, ]
factor_cols = factor_cols[factor_cols <= 100]
dt = dt[, (names(factor_cols)) := lapply(.SD, as.factor), .SD = names(factor_cols)]

# change IDate to date, because of error
# Assertion on 'feature types' failed: Must be a subset of
# {'logical','integer','numeric','character','factor','ordered','POSIXct'},
# but has additional elements {'IDate'}.
dt[, date := as.POSIXct(date, tz = "UTC")]

# Sort
# this returns error on HPC. Some problem with memory
setorder(dt, date)

# Final checks
dt[, max(date)]

# Save
fwrite(dt, "/home/sn/data/strategies/ml_ohlcv/data.csv")

# Save to padobran
# scp /home/sn/data/strategies/ml_ohlcv/data.csv padobran:/home/jmaric/ml_ohlcv/data.csv
