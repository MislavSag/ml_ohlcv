library(data.table)
library(mlr3batchmark)
library(batchtools)
library(fs)
library(PerformanceAnalytics)
library(AzureStor)
library(ggplot2)
library(matrixStats)


# set up
if (Sys.info()["user"] == "Mislav") {
  PATH = "F:/strategies/mlohlcv/padobran"
}

# import results
# reg = loadRegistry(PATH)
# ids_done = findDone(reg = reg)
# system.time({
#   results = mlr3batchmark::reduceResultsBatchmark(ids_done[1:40], TRUE, reg)
# })

# creds
BLOBENDPOINT = storage_endpoint(Sys.getenv("ENDPOINT"), key=Sys.getenv("KEY"))

# load registry
reg = loadRegistry(PATH, work.dir=PATH)

# Used memory
reg$status[!is.na(mem.used)]
reg$status[, max(mem.used, na.rm = TRUE)]

# Done jobs
results_files = fs::path_ext_remove(fs::path_file(dir_ls(fs::path(PATH, "results"))))
ids_done = findDone(reg=reg)
ids_done = ids_done[job.id %in% results_files]
ids_notdone = findNotDone(reg=reg)

# Reasons for not done:
# 1. Catbootst could not be found
# 2. Memory limit

# get results
tabs = batchtools::getJobTable(ids_done, reg = reg)[
  , c("job.id", "job.name", "repl", "prob.pars", "algo.pars"), with = FALSE]
predictions_meta = cbind.data.frame(
  id = tabs[, job.id],
  task = vapply(tabs$prob.pars, `[[`, character(1L), "task_id"),
  learner = gsub(".*regr.|.tuned", "", vapply(tabs$algo.pars, `[[`, character(1L), "learner_id")),
  cv = gsub("custom_|_.*", "", vapply(tabs$prob.pars, `[[`, character(1L), "resampling_id")),
  fold = gsub("custom_\\d+_", "", vapply(tabs$prob.pars, `[[`, character(1L), "resampling_id"))
)
predictions_l = lapply(unlist(ids_done), function(id_) {
  # id_ = 10035
  x = tryCatch({readRDS(fs::path(PATH, "results", id_, ext = "rds"))},
               error = function(e) NULL)
  if (is.null(x)) {
    print(id_)
    return(NULL)
  }
  x["id"] = id_
  x
})
predictions = lapply(predictions_l, function(x) {
  cbind.data.frame(
    id = x$id,
    row_ids = x$prediction$test$row_ids,
    truth = x$prediction$test$truth,
    response = x$prediction$test$response
  )
})
predictions = rbindlist(predictions)
predictions = merge(predictions_meta, predictions, by = "id")
predictions = as.data.table(predictions)

# import tasks
tasks_files = dir_ls(fs::path(PATH, "problems"))
tasks = lapply(tasks_files, readRDS)
names(tasks) = lapply(tasks, function(t) t$data$id)
tasks

# add backend to predictions
backend_l = lapply(tasks, function(tsk_) {
  x = tsk_$data$backend$data(1:tsk_$data$nrow,
                             c("symbol", "date", "..row_id"))
  setnames(x, c("symbol", "date", "row_ids"))
  x
})
backends = rbindlist(backend_l, fill = TRUE)

# merge predictions and backends
predictions = backends[, .(symbol, date, row_ids)][predictions, on = c("symbol" = "task", "row_ids")]

# measures
# source("AdjLoss2.R")
# source("PortfolioRet.R")
# mlr_measures$add("linex", finautoml::Linex)
# mlr_measures$add("adjloss2", AdjLoss2)
# mlr_measures$add("portfolio_ret", PortfolioRet)

# merge backs and predictions
predictions[, date := as.Date(date)]


# PREDICTIONS RESULTS -----------------------------------------------------
# remove dupliactes - keep firt
dt = unique(predictions, by = c("row_ids", "date", "symbol", "learner", "cv"))

# remove na value
dt = na.omit(dt)

# predictions
dt[, `:=`(
  truth_sign = as.factor(sign(ifelse(truth == 0, 1, truth))),
  response_sign = as.factor(sign(response))
)]
dt[truth_sign == 0]

# classification measures across ids
measures = function(t, res) {
  list(acc   = mlr3measures::acc(t, res),
       fbeta = mlr3measures::fbeta(t, res, positive = "1"),
       tpr   = mlr3measures::tpr(t, res, positive = "1"),
       tnr   = mlr3measures::tnr(t, res, positive = "1"))
}
dt[, measures(truth_sign, response_sign), by = c("cv")]
dt[, measures(truth_sign, response_sign), by = c("symbol")]
dt[, measures(truth_sign, response_sign), by = c("learner")]
dt[, measures(truth_sign, response_sign), by = c("cv", "symbol")]
dt[, measures(truth_sign, response_sign), by = c("cv", "learner")]
dt[, measures(truth_sign, response_sign), by = c("symbol", "learner")]
# predictions[, measures(truth_sign, response_sign), by = c("cv", "task", "learner")][order(V1)]

# prediction to wide format
dtw = dcast(dt, symbol + date + truth + truth_sign ~ learner, value.var = "response")

# ensambles
cols = colnames(dtw)
cols = cols[which(cols == "earth"):ncol(dtw)]
p = dtw[, ..cols]
pm = as.matrix(p)
dtw = cbind(dtw, mean_resp = rowMeans(p, na.rm = TRUE))
dtw = cbind(dtw, median_resp = rowMedians(pm, na.rm = TRUE))
dtw = cbind(dtw, sum_resp = rowSums2(pm, na.rm = TRUE))
dtw = cbind(dtw, iqrs_resp = rowIQRs(pm, na.rm = TRUE))
dtw = cbind(dtw, sd_resp = rowMads(pm, na.rm = TRUE))
dtw = cbind(dtw, q9_resp = rowQuantiles(pm, probs = 0.9, na.rm = TRUE))
dtw = cbind(dtw, max_resp = rowMaxs(pm, na.rm = TRUE))
dtw = cbind(dtw, min_resp = rowMins(pm, na.rm = TRUE))
dtw = cbind(dtw, all_buy = rowAlls(pm >= 0, na.rm = TRUE))
dtw = cbind(dtw, all_sell = rowAlls(pm < 0, na.rm = TRUE))
dtw = cbind(dtw, sum_buy = rowSums2(pm >= 0, na.rm = TRUE))
dtw = cbind(dtw, sum_sell = rowSums2(pm < 0, na.rm = TRUE))
dtw

# results by ensamble statistics for classification measures
calculate_measures = function(t, res) {
  list(acc       = mlr3measures::acc(t, res),
       fbeta     = mlr3measures::fbeta(t, res, positive = "1"),
       tpr       = mlr3measures::tpr(t, res, positive = "1"),
       precision = mlr3measures::precision(t, res, positive = "1"),
       tnr       = mlr3measures::tnr(t, res, positive = "1"),
       npv       = mlr3measures::npv(t, res, positive = "1"))
}
dtw[, calculate_measures(truth_sign, as.factor(sign(mean_resp))), by = symbol]
dtw[, calculate_measures(truth_sign, as.factor(sign(median_resp))), by = symbol]
dtw[, calculate_measures(truth_sign, as.factor(sign(sum_resp))), by = symbol]
dtw[, calculate_measures(truth_sign, factor(ifelse(sum_buy > 8, 1, -1), levels = c(-1, 1)))]

# performance by returns
cols = colnames(dtw)
cols = cols[which(cols == "bart"):which(cols == "sum_resp")]
cols = c("symbol", cols)
melt(na.omit(dtw[, ..cols]), id.vars = "symbol")[value > 0, sum(value),
                                                 by = .(symbol, variable)][order(V1)]
melt(na.omit(dtw[, ..cols]), id.vars = "symbol")[value > 0 & value < 2, sum(value),
                                                 by = .(symbol, variable)][order(V1)]

# Save to azure for QC backtest
cont = storage_container(BLOBENDPOINT, "qc-backtest")
file_name_ =  paste0("mlohlcv.csv")
qc_data = na.omit(copy(dtw)[, rsm := NULL])
qc_data = unique(qc_data, by = c("symbol", "date"))
setorder(qc_data, date, symbol)
qc_data[, .(min_date = min(date), max_date = max(date))]
qc_data[symbol == "tlt"]
storage_write_csv(qc_data, cont, file_name_)


# BACKTEST ----------------------------------------------------------------
# Import symbols data
prices = fread("F:/strategies/mlohlcv/ml-ohlcv-20240731.csv",
               select = c("symbol", "date", "close"))
setorder(prices, symbol, date)
prices[, returns := close / shift(close) - 1, by = symbol]
prices = na.omit(prices)

# Merge predictions with prices
dtb = dtw[prices, on = c("symbol", "date")]
dtb[, rsm := NULL]
dtb = na.omit(dtb)

# Signal creation
cols_responses = colnames(dtb)[5:13]
dtb[, paste0(cols_responses, "_ret") := lapply(.SD, function(x) ifelse(shift(x) > 0, 1, 0) * returns),
    .SDcols = cols_responses]
dtb = dtb[, .SD, .SDcols = c("symbol", "date", "returns", paste0(cols_responses, "_ret"))]
cols = colnames(dtb)[-1]
dtb1 = as.xts.data.table(na.omit(dtb[symbol == "spy", .SD, .SDcols = colnames(dtb)[-1]]))
dtb2 = as.xts.data.table(na.omit(dtb[symbol == "tlt", .SD, .SDcols = colnames(dtb)[-1]]))
PerformanceAnalytics::charts.PerformanceSummary(dtb1)
PerformanceAnalytics::charts.PerformanceSummary(dtb2)
