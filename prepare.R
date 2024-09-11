library(data.table)
library(arrow)
library(dplyr)
library(finfeatures)


# SET UP ------------------------------------------------------------------
# Parameters
symbols = c("tlt", "spy")

# Globals
PATH = "F:/strategies/mlohlcv"
# get_ohlcv_by_symbols = function(path, symbols) {
#   con = dbConnect(duckdb::duckdb())
#   symbols_string = paste(sprintf("'%s'", symbols), collapse=", ")
#   query = sprintf("
#   SELECT *
#   FROM '%s'
#   WHERE Symbol IN (%s)
# ", path, symbols_string)
#   prices_dt = dbGetQuery(con, query)
#   dbDisconnect(con, shutdown = TRUE)
#   setDT(prices_dt)
#   return(prices_dt)
# }
get_ohlcv_by_symbols = function(tag, symbols) {
  path_ = paste0("F:/data/equity/us/predictors_daily/", tag)
  ds = open_dataset(path_, format = "parquet") %>%
    filter(symbol %in% symbols) %>%
    mutate(symbol = as.character(symbol)) %>%
    collect()
  setDT(ds)
  setorder(ds, symbol, date)
  setcolorder(ds, "symbol", before = 1)
  return(ds)
}


# DATA --------------------------------------------------------------------
# OHLCV Prices
prices = get_ohlcv_by_symbols("F:/strategies/momentum/prices.csv", symbols)


# f = list.files("F:/data/equity/us/predictors_daily/fracdiff")
# ft = grepl("symbol=[a-z]", f)
# fn = nchar(f)
# all(fn > 7)
# nchar
# nchar("symbol=")
# all(ft)

# Predictors
# https://github.com/duckdb/duckdb/issues/4295
system.time({backcusum = get_ohlcv_by_symbols("backcusum", symbols)})
system.time({exuber = get_ohlcv_by_symbols("exuber", symbols)})
system.time({forecasts = get_ohlcv_by_symbols("forecasts", symbols)})
system.time({fracdiff = get_ohlcv_by_symbols("fracdiff", symbols)})
system.time({theftpy = get_ohlcv_by_symbols("theftpy", symbols)})
system.time({theftr = get_ohlcv_by_symbols("theftr", symbols)})
system.time({tsfeatures = get_ohlcv_by_symbols("tsfeatures", symbols)})

# Generate other Ohlcv predictors
ohlcv = Ohlcv$new(prices)
ohlcv_features_daily = OhlcvFeaturesDaily$new(
  at = NULL,
  windows = c(5, 10, 22, 44, 66, 132, 252, 504, 756),
  quantile_divergence_window = c(22, 44, 66, 132, 252, 504, 756)
)
ohlcv_predictors = ohlcv_features_daily$get_ohlcv_features(ohlcv$X)

# Merge prices and all predictors
dt = Reduce(
  function(x, y) merge(x, y, by = c("symbol", "date"), all = TRUE),
  list(ohlcv_predictors, exuber, backcusum, forecasts, fracdiff)
)

# Inspect
dim(dt)
head(colnames(dt), 20)
tail(colnames(dt), 20)


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
  "symbol", "date", "liquid_500", "liquid_200", "liquid_100", "open", "high",
  "low", "close", "volume", "close_raw", "returns", target_columns)
cols_features = setdiff(colnames(dt), cols_non_features)
cols = c(cols_non_features, cols_features)


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

# Final checks
dt[, max(date)]

# Save features
last_date = strftime(dt[, max(date)], "%Y%m%d")
file_name = paste0("ml-ohlcv-", last_date, ".csv")
file_name_local = fs::path(PATH, file_name)
fwrite(dt, file_name_local)

# Send data to padobran mannually


