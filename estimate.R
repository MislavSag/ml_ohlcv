library(data.table)
library(mlr3verse)
library(lubridate)
library(finautoml)
library(paradox)
library(batchtools)
library(mlr3batchmark)


# SETUP -------------------------------------------------------------------
# Parameters
ncpus = 4
cv = "stack" # "stack" or "parallel"


# DATA --------------------------------------------------------------------
# Read data
if (interactive()) {
  dt = fread("/home/sn/data/strategies/ml_ohlcv/data.csv")
} else {
  dt = fread("data.csv")
}

# Define predictors, targets and ids
cols_target = colnames(dt)[grep("target", colnames(dt))]
cols_ids = c("symbol", "date", cols_target)
cols_remove = c("open", "high", "low", "close", "volume", "close_raw","returns")
cols_predictors = setdiff(colnames(dt), c(cols_ids, cols_remove))

# int64 to integer
cols = dt[ , colnames(.SD), .SDcols = bit64::is.integer64]
dt[ , (cols) := lapply(.SD, as.numeric), .SDcols = cols]


# ADD PIPELINES -----------------------------------------------------------
mlr_pipeops$add("uniformization", finautoml::PipeOpUniform)
# mlr_pipeops$add("winsorize", PipeOpWinsorize)
mlr_pipeops$add("winsorizesimple", finautoml::PipeOpWinsorizeSimple)
# mlr_pipeops$add("winsorizesimplegroup", PipeOpWinsorizeSimpleGroup)
mlr_pipeops$add("dropna", finautoml::PipeOpDropNA)
mlr_pipeops$add("dropnacol", finautoml::PipeOpDropNACol)
mlr_pipeops$add("dropcorr", finautoml::PipeOpDropCorr)
# mlr_pipeops$add("pca_explained", PipeOpPCAExplained)
mlr_pipeops$add("filter_target", finautoml::PipeOpFilterRegrTarget)
mlr_pipeops$add("filter_rows", finautoml::PipeOpFilterRows)
mlr_filters$add("gausscov_f1st", finautoml::FilterGausscovF1st)
# mlr_filters$add("gausscov_f3st", FilterGausscovF3st)
mlr_measures$add("linex", finautoml::Linex)
mlr_measures$add("adjloss2", finautoml::AdjLoss2)
# mlr_measures$add("portfolio_ret", PortfolioRet)


# LEARNERS ----------------------------------------------------------------
# graph templates
gr = gunion(list(
  po("nop", id = "nop_union_pca"),
  po("pca", center = FALSE, rank. = 50),
  po("ica", n.comp = 10)
  # po("ica", n.comp = 10) # Error in fastICA::fastICA(as.matrix(dt), n.comp = 10L, method = "C") : data must be matrix-conformal This happened PipeOp ica's $train()
)) %>>% po("featureunion", id = "feature_union_pca")
filter_target_gr = po("branch",
                      options = c("nop_filter_target", "filter_target_select"),
                      id = "filter_target_branch") %>>%
  gunion(list(
    po("nop", id = "nop_filter_target"),
    po("filter_target", id = "filter_target_id")
  )) %>>%
  po("unbranch", id = "filter_target_unbranch")
filter_rows_branch = po("branch",
                         options = c("nop_filter_rows", "filter_rows"),
                         id = "filter_rows_branch") %>>%
  gunion(list(
    po("nop", id = "nop_filter_rows"),
    po("filter_rows", id = "filter_rows_id", phase = "train")
  )) %>>%
  po("unbranch", id = "filter_rows_unbranch")
# create mlr3 graph that takes 3 filters and union predictors from them
filters_ = list(
  po("filter", flt("disr"), filter.nfeat = 5),
  po("filter", flt("jmim"), filter.nfeat = 5),
  po("filter", flt("jmi"), filter.nfeat = 5),
  po("filter", flt("mim"), filter.nfeat = 5),
  po("filter", flt("mrmr"), filter.nfeat = 5),
  po("filter", flt("njmim"), filter.nfeat = 5),
  po("filter", flt("cmim"), filter.nfeat = 5),
  po("filter", flt("carscore"), filter.nfeat = 5), # UNCOMMENT LATER< SLOWER SO COMMENTED FOR DEVELOPING
  po("filter", flt("information_gain"), filter.nfeat = 5),
  po("filter", filter = flt("relief"), filter.nfeat = 5),
  po("filter", filter = flt("gausscov_f1st"), p0 = 0.00001, filter.cutoff = 0)
  # po("filter", mlr3filters::flt("importance", learner = mlr3::lrn("classif.rpart")), filter.nfeat = 10, id = "importance_1"),
  # po("filter", mlr3filters::flt("importance", learner = lrn), filter.nfeat = 10, id = "importance_2")
)
graph_filters = gunion(filters_) %>>%
  po("featureunion", length(filters_), id = "feature_union_filters")

graph_template =
  filter_rows_branch %>>%
  filter_target_gr %>>%
  # Hyperband tuning
  po("subsample") %>>% # uncomment this for hyperparameter tuning
  # Simple preprocessing
  po("dropnacol", id = "dropnacol", cutoff = 0.05) %>>%
  po("dropna", id = "dropna") %>>%
  po("removeconstants", id = "removeconstants_1", ratio = 0)  %>>%
  po("fixfactors", id = "fixfactors") %>>%
  # Winsorization
  po("branch", options = c("winsoriaztion", "winsoriaztion_nop"), id = "winsorization_branch") %>>%
  gunion(list(po("winsorizesimple", probs_low = 0.01, probs_high = 0.99, na_rm = TRUE),
              po("nop")
  )) %>>%
  po("unbranch", id = "winsoriaztion_unbranch") %>>%
  # po("winsorizesimple", id = "winsorizesimple", probs_low = 0.01, probs_high = 0.99, na_rm = TRUE) %>>%
  # po("winsorizesimplegroup", group_var = "weekid", id = "winsorizesimplegroup", probs_low = 0.01, probs_high = 0.99, na.rm = TRUE) %>>%
  po("removeconstants", id = "removeconstants_2", ratio = 0)  %>>%
  # Drop correlated columns
  po("dropcorr", id = "dropcorr", cutoff = 0.99) %>>%
  # po("uniformization") %>>%
  # scale branch
  po("branch", options = c("uniformization", "scale"), id = "scale_branch") %>>%
  gunion(list(po("uniformization"),
              po("scale")
  )) %>>%
  po("unbranch", id = "scale_unbranch") %>>%
  po("dropna", id = "dropna_v2") %>>%
  # add pca columns
  gr %>>%
  # filters
  # OLD BARNCH WAY
  # po("branch", options = c("jmi", "gausscovf1", "gausscovf3"), id = "filter_branch") %>>%
  # gunion(list(po("filter", filter = flt("jmi"), filter.nfeat = 25),
  #             # po("filter", filter = flt("relief"), filter.nfeat = 25),
  #             po("filter", filter = flt("gausscov_f1st"), p0 = 0.2, kmn=10, filter.cutoff = 0),
  #             po("filter", filter = flt("gausscov_f3st"), p0 = 0.2, kmn=10, m = 1, filter.cutoff = 0)
  # )) %>>%
  # po("unbranch", id = "filter_unbranch") %>>%
  # OLD BARNCH WAY
  # filters union
  graph_filters %>>%
  # modelmatrix
  # po("branch", options = c("nop_interaction", "modelmatrix"), id = "interaction_branch") %>>%
  # gunion(list(
  #   po("nop", id = "nop_interaction"),
  #   po("modelmatrix", formula = ~ . ^ 2))) %>>%
  # po("unbranch", id = "interaction_unbranch") %>>%
  po("removeconstants", id = "removeconstants_3", ratio = 0)
plot(graph_template)

# Inspect filtering columns
# pattterns: back.*Two.*Rej,
if (interactive()) {
  int_cols = dt[, vapply(.SD, function(x) uniqueN(x) < 4, FUN.VALUE = logical(1))]
  int_cols = names(int_cols[int_cols])
  cols = colnames(dt)[grepl("tsfreshValuesAggLinearTrend*", colnames(dt))]
  x = dt[, .SD, .SDcols = c("symbol", "date", cols)]
  x = na.omit(melt(x, id.vars = c("symbol", "date")))[
    , nrow(.SD[value == 1]) / nrow(.SD), by = variable]
  x[V1 %between% c(0.05, 0.25)]
}

# hyperparameters template
as.data.table(graph_template$param_set)[1:100]
gausscov_sp = as.character(seq(0.2, 0.8, by = 0.1))
winsorize_sp =  c(0.999, 0.99, 0.98, 0.97, 0.90, 0.8)
search_space_template = ps(
  # filter target
  filter_target_branch.selection = p_fct(levels = c("nop_filter_target", "filter_target_select")),
  filter_target_id.q = p_fct(levels = c("0.4", "0.3", "0.2"),
                             trafo = function(x, param_set) return(as.double(x)),
                             depends = filter_target_branch.selection == "filter_target_select"),
  # subsample for hyperband
  subsample.frac = p_dbl(0.7, 1, tags = "budget"), # commencement this if we want to use hyperband optimization
  # preprocessing
  dropcorr.cutoff = p_fct(c("0.80", "0.90", "0.95", "0.99"),
                          trafo = function(x, param_set) return(as.double(x))),
  winsorization_branch.selection = p_fct(levels = c("winsoriaztion", "winsoriaztion_nop")),
  winsorization.probs_high = p_fct(
    as.character(winsorize_sp),
    trafo = function(x, param_set) return(as.double(x)),
    depends = winsorization_branch.selection == "winsoriaztion"),
  winsorization.probs_low = p_fct(
    as.character(1 - winsorize_sp),
    trafo = function(x, param_set) return(as.double(x)),
    depends = winsorization_branch.selection == "winsoriaztion"),
  # scaling
  scale_branch.selection = p_fct(levels = c("uniformization", "scale")),
  # Filter rows
  filter_rows_branch.selection = p_fct(levels = c("nop_filter_rows", "filter_rows")),
  filter_rows_id.filter_formula = p_fct(
    levels = as.character(1:3),
    depends = filter_rows_branch.selection == "filter_rows"),
  .extra_trafo = function(x, param_set) {
    if (x$filter_rows_id.filter_formula == "1") {
      x$filter_rows_id.filter_formula = as.formula("~ backcusum66TwoSided2BackcusumRejections10 == 1")
    } else if (x$filter_rows_id.filter_formula == "2") {
      x$filter_rows_id.filter_formula = as.formula("~ backcusum66TwoSided2BackcusumRejections100 == 1")
    } else if (x$filter_rows_id.filter_formula == "3") {
      x$filter_rows_id.filter_formula = as.formula("~ backcusum66TwoSided2BackcusumRejections50 == 1")
    }
    return(x)
  }
)

if (interactive()) {
  # show all combinations from search space, like in grid
  sp_grid = generate_design_grid(search_space_template, 1)
  sp_grid = sp_grid$data
  sp_grid

  # check ids of nth cv sets
  train_ids = custom_cvs[[1]]$inner$instance$train[[1]]

  # help graph for testing preprocessing
  preprocess_test = function() {
    task_ = tasks[[1]]$clone()
    nr = task_$nrow
    rows_ = (nr-10000):nr
    # task_$filter(rows_)
    task_$filter(train_ids)
    # dates = task_$backend$data(rows_, "date")
    # print(dates[, min(date)])
    # print(dates[, max(date)])
    gr_test = graph_template$clone()
    # gr_test$param_set$set_values(
    #   filter_rows_id.filter_formula = as.formula("~ backcusum66TwoSided2BackcusumRejections50 == 1"),
    #   filter_rows_branch.selection = "filter_rows",
    #   winsorization_branch.selection = "winsoriaztion",
    #   filter_target_branch.selection = "filter_target_select"
    # )
    gr_test$param_set$set_values(
      filter_rows_branch.selection = "nop_filter_rows",
      winsorization_branch.selection = "winsoriaztion",
      filter_target_branch.selection = "nop_filter_target"
    )
    return(gr_test$train(task_))
  }

  # test graph preprocesing
  system.time({test_default = preprocess_test()})

  # TEst gausscov
}

# random forest graph
graph_rf = graph_template %>>%
  po("learner", learner = lrn("regr.ranger"))
graph_rf = as_learner(graph_rf)
as.data.table(graph_rf$param_set)[, .(id, class, lower, upper, levels)]
search_space_rf = c(
  search_space_template,
  ps(
    regr.ranger.max.depth  = p_int(1, 15),
    regr.ranger.replace    = p_lgl(),
    regr.ranger.mtry.ratio = p_dbl(0.3, 1),
    regr.ranger.num.trees  = p_int(10, 2000),
    regr.ranger.splitrule  = p_fct(levels = c("variance", "extratrees"))
  )
)

# xgboost graph
graph_xgboost = graph_template %>>%
  po("learner", learner = lrn("regr.xgboost"))
plot(graph_xgboost)
graph_xgboost = as_learner(graph_xgboost)
as.data.table(graph_xgboost$param_set)[grep("depth", id), .(id, class, lower, upper, levels)]
search_space_xgboost = c(
  search_space_template,
  ps(
    regr.xgboost.alpha     = p_dbl(0.001, 100, logscale = TRUE),
    regr.xgboost.max_depth = p_int(1, 20),
    regr.xgboost.eta       = p_dbl(0.0001, 1, logscale = TRUE),
    regr.xgboost.nrounds   = p_int(1, 5000),
    regr.xgboost.subsample = p_dbl(0.1, 1)
  )
)

# gbm graph
graph_gbm = graph_template %>>%
  po("learner", learner = lrn("regr.gbm"))
plot(graph_gbm)
graph_gbm = as_learner(graph_gbm)
as.data.table(graph_gbm$param_set)[, .(id, class, lower, upper, levels)]
search_space_gbm = c(
  search_space_template,
  ps(
    regr.gbm.distribution      = p_fct(levels = c("gaussian", "tdist")),
    regr.gbm.shrinkage         = p_dbl(lower = 0.001, upper = 0.1),
    regr.gbm.n.trees           = p_int(lower = 50, upper = 150),
    regr.gbm.interaction.depth = p_int(lower = 1, upper = 3)
  )
)

# catboost graph
graph_catboost = graph_template %>>%
  po("learner", learner = lrn("regr.catboost"))
graph_catboost = as_learner(graph_catboost)
as.data.table(graph_catboost$param_set)[, .(id, class, lower, upper, levels)]
# https://catboost.ai/en/docs/concepts/parameter-tuning#description10
search_space_catboost = c(
  search_space_template,
  ps(
    regr.catboost.learning_rate   = p_dbl(lower = 0.01, upper = 0.3),
    regr.catboost.depth           = p_int(lower = 4, upper = 10),
    regr.catboost.l2_leaf_reg     = p_int(lower = 1, upper = 5),
    regr.catboost.random_strength = p_int(lower = 0, upper = 3)
  )
)

# cforest graph
graph_cforest = graph_template %>>%
  po("learner", learner = lrn("regr.cforest"))
graph_cforest = as_learner(graph_cforest)
as.data.table(graph_cforest$param_set)[, .(id, class, lower, upper, levels)]
# https://cran.r-project.org/web/packages/partykit/partykit.pdf
search_space_cforest = c(
  search_space_template,
  ps(
    regr.cforest.mtryratio    = p_dbl(0.05, 1),
    regr.cforest.ntree        = p_int(lower = 10, upper = 2000),
    regr.cforest.mincriterion = p_dbl(0.1, 1),
    regr.cforest.replace      = p_lgl()
  )
)

# # gamboost graph
# # Error in eval(predvars, data, env) : object 'adxDx14' not found
# # This happened PipeOp regr.gamboost's $train()
# # In addition: There were 50 or more warnings (use warnings() to see the first 50)
# graph_gamboost = graph_template %>>%
#   po("learner", learner = lrn("regr.gamboost"))
# graph_gamboost = as_learner(graph_gamboost)
# as.data.table(graph_gamboost$param_set)[, .(id, class, lower, upper, levels)]
# search_space_gamboost = search_space_template$clone()
# search_space_gamboost$add(
#   ps(regr.gamboost.mstop       = p_int(lower = 10, upper = 100),
#      regr.gamboost.nu          = p_dbl(lower = 0.01, upper = 0.5),
#      regr.gamboost.baselearner = p_fct(levels = c("bbs", "bols", "btree")))
# )

# kknn graph
graph_kknn = graph_template %>>%
  po("learner", learner = lrn("regr.kknn"))
graph_kknn = as_learner(graph_kknn)
as.data.table(graph_kknn$param_set)[, .(id, class, lower, upper, levels)]
search_space_kknn = c(
  search_space_template,
  ps(
    regr.kknn.k        = p_int(lower = 1, upper = 50, logscale = TRUE),
    regr.kknn.distance = p_dbl(lower = 1, upper = 5),
    regr.kknn.kernel   = p_fct(levels = c("rectangular","optimal", "epanechnikov",
                                          "biweight", "triweight", "cos", "inv",
                                          "gaussian","rank"))
  )
)

# nnet graph
graph_nnet = graph_template %>>%
  po("learner", learner = lrn("regr.nnet", MaxNWts = 50000))
graph_nnet = as_learner(graph_nnet)
as.data.table(graph_nnet$param_set)[, .(id, class, lower, upper, levels)]
search_space_nnet = c(
  search_space_template,
  ps(
    regr.nnet.size  = p_int(lower = 2, upper = 15),
    regr.nnet.decay = p_dbl(lower = 0.0001, upper = 0.1),
    regr.nnet.maxit = p_int(lower = 50, upper = 500)
  )
)

# glmnet graph
graph_glmnet = graph_template %>>%
  po("learner", learner = lrn("regr.glmnet"))
graph_glmnet = as_learner(graph_glmnet)
as.data.table(graph_glmnet$param_set)[, .(id, class, lower, upper, levels)]
search_space_glmnet = c(
  search_space_template,
  ps(
    regr.glmnet.s     = p_int(lower = 5, upper = 30),
    regr.glmnet.alpha = p_dbl(lower = 1e-4, upper = 1, logscale = TRUE)
  )
)

### THIS LEARNER UNSTABLE ####
# ksvm graph
# graph_ksvm = graph_template %>>%
#   po("learner", learner = lrn("regr.ksvm"), scaled = FALSE)
# graph_ksvm = as_learner(graph_ksvm)
# as.data.table(graph_ksvm$param_set)[, .(id, class, lower, upper, levels)]
# search_space_ksvm = ps(
#   # subsample for hyperband
#   subsample.frac = p_dbl(0.3, 1, tags = "budget"), # unccoment this if we want to use hyperband optimization
#   # preprocessing
#   dropcorr.cutoff = p_fct(
#     levels = c("0.80", "0.90", "0.95", "0.99"),
#     trafo = function(x, param_set) {
#       switch(x,
#              "0.80" = 0.80,
#              "0.90" = 0.90,
#              "0.95" = 0.95,
#              "0.99" = 0.99)
#     }
#   ),
#   # dropcorr.cutoff = p_fct(levels = c(0.8, 0.9, 0.95, 0.99)),
#   winsorizesimplegroup.probs_high = p_fct(levels = c(0.999, 0.99, 0.98, 0.97, 0.90, 0.8)),
#   winsorizesimplegroup.probs_low = p_fct(levels = c(0.001, 0.01, 0.02, 0.03, 0.1, 0.2)),
#   # filters
#   filter_branch.selection = p_fct(levels = c("jmi", "relief", "gausscov")),
#   # interaction
#   interaction_branch.selection = p_fct(levels = c("nop_interaction", "modelmatrix")),
#   # learner
#   regr.ksvm.kernel  = p_fct(levels = c("rbfdot", "polydot", "vanilladot",
#                                        "laplacedot", "besseldot", "anovadot")),
#   regr.ksvm.C       = p_dbl(lower = 0.0001, upper = 1000, logscale = TRUE),
#   regr.ksvm.degree  = p_int(lower = 1, upper = 5,
#                             depends = regr.ksvm.kernel %in% c("polydot", "besseldot", "anovadot")),
#   regr.ksvm.epsilon = p_dbl(lower = 0.01, upper = 1)
# )
### THIS LEARNER UNSTABLE ####

# LAST
# lightgbm graph
# [LightGBM] [Fatal] Do not support special JSON characters in feature name.
graph_lightgbm = graph_template %>>%
  po("learner", learner = lrn("regr.lightgbm"))
graph_lightgbm = as_learner(graph_lightgbm)
as.data.table(graph_lightgbm$param_set)[grep("sample", id), .(id, class, lower, upper, levels)]
search_space_lightgbm = c(
  search_space_template,
  ps(
    regr.lightgbm.max_depth     = p_int(lower = 2, upper = 10),
    regr.lightgbm.learning_rate = p_dbl(lower = 0.001, upper = 0.3),
    regr.lightgbm.num_leaves    = p_int(lower = 10, upper = 100)
  )
)

# earth graph
graph_earth = graph_template %>>%
  po("learner", learner = lrn("regr.earth"))
graph_earth = as_learner(graph_earth)
as.data.table(graph_earth$param_set)[grep("sample", id), .(id, class, lower, upper, levels)]
search_space_earth = c(
  search_space_template,
  ps(
    regr.earth.degree  = p_int(lower = 1, upper = 4),
    # regr.earth.penalty = p_int(lower = 1, upper = 5),
    regr.earth.nk      = p_int(lower = 50, upper = 250)
  )
)

# rsm graph
# TODO It need lots of memory ?
graph_rsm = graph_template %>>%
  po("learner", learner = lrn("regr.rsm"))
plot(graph_rsm)
graph_rsm = as_learner(graph_rsm)
as.data.table(graph_rsm$param_set)[, .(id, class, lower, upper, levels)]
search_space_rsm = c(
  search_space_template,
  ps(regr.rsm.modelfun = p_fct(levels = c("FO", "TWI", "SO")))
)
# Error in fo[, i] * fo[, j] : non-numeric argument to binary operator
# This happened PipeOp regr.rsm's $train()
# Calls: lapply ... resolve.list -> signalConditionsASAP -> signalConditions
# In addition: Warning message:
# In bbandsDn5:volumeDownUpRatio :
#   numerical expression has 4076 elements: only the first used
# This happened PipeOp regr.rsm's $train()

# BART graph
graph_bart = graph_template %>>%
  po("learner", learner = lrn("regr.bart"))
graph_bart = as_learner(graph_bart)
as.data.table(graph_bart$param_set)[, .(id, class, lower, upper, levels)]
search_space_bart = c(
  search_space_template,
  ps(
    regr.bart.k      = p_dbl(lower = 1, upper = 10),
    regr.bart.numcut = p_int(lower = 10, upper = 200),
    regr.bart.ntree  = p_int(lower = 50, upper = 500)
  )
)
# chatgpt returns this
# n_chains = p_int(lower = 1, upper = 5),
# m_try = p_int(lower = 1, upper = 13),
# nu = p_dbl(lower = 0.1, upper = 10),
# alpha = p_dbl(lower = 0.01, upper = 1),
# beta = p_dbl(lower = 0.01, upper = 1),
# burn = p_int(lower = 10, upper = 100),
# iter = p_int(lower = 100, upper = 1000)

# Threads
set_threads(graph_rf, n = ncpus)
set_threads(graph_xgboost, n = ncpus)
set_threads(graph_bart, n = ncpus)
# set_threads(graph_ksvm, n = threads) # unstable
set_threads(graph_nnet, n = ncpus)
set_threads(graph_kknn, n = ncpus)
set_threads(graph_lightgbm, n = ncpus)
set_threads(graph_earth, n = ncpus)
set_threads(graph_gbm, n = ncpus)
set_threads(graph_rsm, n = ncpus)
set_threads(graph_catboost, n = ncpus)
set_threads(graph_glmnet, n = ncpus)
set_threads(graph_cforest, n = ncpus)


# CV ----------------------------------------------------------------------
# Cross validation
create_custom_rolling_windows = function(task,
                                         duration_unit = "week", # can be day, week, month
                                         train_duration,
                                         gap_duration,
                                         tune_duration,
                                         test_duration) {
  # Debug
  # task = tasks[[2]]$clone()
  # train_duration = 8 * 5 * 12
  # gap_duration = 1
  # tune_duration = 5 * 3
  # test_duration = 1

  # Function to convert durations to the appropriate period
  convert_duration = function(duration) {
    if (duration_unit == "day") {
      days(duration)
    } else if (duration_unit == "week") {
      weeks(duration)
    } else { # defaults to months if not weeks
      months(duration)
    }
  }

  # Define row ids
  data_ = task$backend$data(cols = c("date", "..row_id"), rows = 1:task$nrow)
  setnames(data_, "..row_id", "row_id")
  stopifnot(all(task$row_ids == data_[, row_id]))

  # Ensure date is in Date format
  data_[, date_col := as.Date(date)]

  # Initialize start and end dates based on duration unit
  start_date =
    if (duration_unit == "day") {
      data_[, min(date_col)]
    } else if (duration_unit == "week") {
      floor_date(data_[, min(date_col)], "week")
    } else {
      floor_date(data_[, min(date_col)], "month")
    }
  end_date =
    if (duration_unit == "day") {
      ceiling_date(data_[, max(date_col)], "week") - days(1)
    } else if (duration_unit == "week") {
      ceiling_date(data_[, max(date_col)], "week") - days(1)
    } else {
      ceiling_date(data_[, max(date_col)], "month") - days(1)
    }

  # Initialize folds list
  folds = list()

  while (start_date < end_date) {
    train_end = start_date %m+% convert_duration(train_duration) - days(1)
    gap1_end  = train_end %m+% convert_duration(gap_duration)
    tune_end  = gap1_end %m+% convert_duration(tune_duration)
    gap2_end  = tune_end %m+% convert_duration(gap_duration)
    test_end  = gap2_end %m+% convert_duration(test_duration) - days(1)

    # Ensure the fold does not exceed the data range
    if (test_end > end_date) {
      break
    }

    # Extracting indices for each set
    train_indices = data_[date_col %between% c(start_date, train_end), row_id]
    tune_indices  = data_[date_col %between% c(gap1_end + days(1), tune_end), row_id]
    test_indices  = data_[date_col %between% c(gap2_end + days(1), test_end), row_id]

    folds[[length(folds) + 1]] <- list(train = train_indices, tune = tune_indices, test = test_indices)

    # Update the start date for the next fold
    start_date = if (duration_unit == "week") {
      start_date %m+% weeks(1)
    } else {
      start_date %m+% months(1)
    }
  }

  # Prepare sets for inner and outer resampling
  train_sets = lapply(folds, function(fold) fold$train)
  tune_sets  = lapply(folds, function(fold) fold$tune)
  test_sets  = lapply(folds, function(fold) fold$test)

  # Combine train and tune sets for outer resampling
  inner_sets = lapply(seq_along(train_sets), function(i) {
    c(train_sets[[i]], tune_sets[[i]])
  })

  # Instantiate custom inner resampling (train: train, test: tune)
  custom_inner = rsmp("custom", id = paste0(task$id, "-inner"))
  custom_inner$instantiate(task, train_sets, tune_sets)

  # Instantiate custom outer resampling (train: train+tune, test: test)
  custom_outer = rsmp("custom", id = paste0(task$id, "-outer"))
  custom_outer$instantiate(task, inner_sets, test_sets)

  return(list(outer = custom_outer, inner = custom_inner))
}

# Parallel CV
if (cv == "parallel") {
  # Create tasks for every symbol
  str(dt)
  tasks = lapply(dt[, unique(symbol)], function(symbol_) {
    print(symbol_)
    task_ = as_task_regr(
      dt[symbol == symbol_, .SD, .SDcols = c(cols_ids, cols_predictors)],
      id = symbol_,
      target = cols_target)
    task_$col_roles$feature = setdiff(task_$col_roles$feature, cols_ids)
    return(task_)
  })

  # Check
  tasks[[1]]$data(cols = c("symbol", "date"))
  tasks[[2]]$data(cols = c("symbol", "date"))
} else if (cv == "stack") {
  # Create tasks for every symbol
  tasks = as_task_regr(dt[, .SD, .SDcols = c(cols_ids, cols_predictors)],
                       id = "stacked",
                       target = cols_target)
  tasks$col_roles$feature = setdiff(tasks$col_roles$feature, cols_ids)
  tasks = list(tasks) # This is just to be a list as in parallel tasks

  # Check
  tasks[[1]]$data(cols = c("symbol", "date"))
}

# Create CVS's for every task
custom_cvs = lapply(tasks, function(task_) {
  create_custom_rolling_windows(
    task = task_$clone(),
    duration_unit = "week",
    train_duration = 52*10,
    gap_duration = 1,
    tune_duration = 5*3,
    test_duration = 1
  )
})

# Check
# visualize test
if (interactive()) {
  library(ggplot2)
  library(patchwork)
  prepare_cv_plot = function(x, set = "train") {
    x = lapply(x, function(x) data.table(ID = x))
    x = rbindlist(x, idcol = "fold")
    x[, fold := as.factor(fold)]
    x[, set := as.factor(set)]
    x[, ID := as.numeric(ID)]
  }
  plot_cv = function(cv, n = 30) {
    # cv = custom_cvs[[1]]
    print(cv)
    cv_test_inner = cv$inner
    cv_test_outer = cv$outer

    # prepare train, tune and test folds
    train_sets = cv_test_inner$instance$train[1:n]
    train_sets = prepare_cv_plot(train_sets)
    tune_sets = cv_test_inner$instance$test[1:n]
    tune_sets = prepare_cv_plot(tune_sets, set = "tune")
    test_sets = cv_test_outer$instance$test[1:n]
    test_sets = prepare_cv_plot(test_sets, set = "test")
    dt_vis = rbind(train_sets[seq(1, nrow(train_sets), 10)],
                   tune_sets[seq(1, nrow(tune_sets), 5)],
                   test_sets[seq(1, nrow(test_sets), 1)])
    substr(colnames(dt_vis), 1, 1) = toupper(substr(colnames(dt_vis), 1, 1))
    ggplot(dt_vis, aes(x = Fold, y = ID, color = Set)) +
      geom_point() +
      theme_minimal() +
      coord_flip() +
      labs(x = "", y = '',
           title = paste0(gsub("-.*", "", cv_test_outer$id)))
  }
  plots = lapply(custom_cvs, plot_cv, n = 30)
  wp = wrap_plots(plots)
  ggsave("plot_cv.png", plot = wp, width = 10, height = 8, dpi = 300)
  print(wp)
}


# DESIGNS -----------------------------------------------------------------
# Create design that will be used for all learners
designs_parallel_l = lapply(seq_along(custom_cvs), function(j) {
  # debug
  # j = 1

  # extract curernt cv
  cv_ = custom_cvs[[j]]

  # get cv inner object
  cv_inner = cv_$inner
  cv_outer = cv_$outer
  cat("Number of iterations fo cv inner is ", cv_inner$iters, "\n")

  # Define index
  if (interactive()) {
    indecies_ = 1:2
  } else {
    indecies_ = 1:cv_inner$iters
  }

  designs_cv_l = lapply(indecies_, function(i) {
    # debug
    # i = 1

    # with new mlr3 version I have to clone
    task_inner = tasks[[j]]$clone()
    task_inner$filter(c(cv_inner$train_set(i), cv_inner$test_set(i)))

    # inner resampling
    custom_ = rsmp("custom")
    custom_$id = paste0("custom_", cv_inner$id, cv_inner$iters, "_", i)
    custom_$instantiate(task_inner,
                        list(cv_inner$train_set(i)),
                        list(cv_inner$test_set(i)))

    # objects for all autotuners
    measure_ = msr("linex")
    tuner_   = tnr("hyperband", eta = 4)
    # q: What does above eta parameter mean?
    # a: The eta parameter is the reduction factor of the number of configurations
    #     and is used to reduce the number of configurations in each iteration.
    #     The number of configurations is reduced by a factor of eta in each iteration.
    #     The default value is 3.
    #     The number of configurations is reduced by a factor of eta in each iteration.
    # q: If I want more detailed search, should I use smaller or larger eta?
    # a: If you want more detailed search, you should use a smaller eta.
    #     The smaller the eta, the more configurations are evaluated.

    # tuner_   = tnr("mbo")
    # term_evals = 20

    # auto tuner rf
    at_rf = auto_tuner(
      tuner = tuner_,
      learner = graph_rf,
      resampling = custom_,
      measure = measure_,
      search_space = search_space_rf,
      terminator = trm("none")
      # term_evals = term_evals
    )

    # auto tuner xgboost
    at_xgboost = auto_tuner(
      tuner = tuner_,
      learner = graph_xgboost,
      resampling = custom_,
      measure = measure_,
      search_space = search_space_xgboost,
      terminator = trm("none")
      # term_evals = term_evals
    )

    # auto tuner BART
    at_bart = auto_tuner(
      tuner = tuner_,
      learner = graph_bart,
      resampling = custom_,
      measure = measure_,
      search_space = search_space_bart,
      terminator = trm("none")
      # term_evals = term_evals
    )

    # auto tuner nnet
    at_nnet = auto_tuner(
      tuner = tuner_,
      learner = graph_nnet,
      resampling = custom_,
      measure = measure_,
      search_space = search_space_nnet,
      terminator = trm("none")
      # term_evals = term_evals
    )

    # auto tuner lightgbm
    at_lightgbm = auto_tuner(
      tuner = tuner_,
      learner = graph_lightgbm,
      resampling = custom_,
      measure = measure_,
      search_space = search_space_lightgbm,
      terminator = trm("none")
      # term_evals = term_evals
    )

    # auto tuner earth
    at_earth = auto_tuner(
      tuner = tuner_,
      learner = graph_earth,
      resampling = custom_,
      measure = measure_,
      search_space = search_space_earth,
      terminator = trm("none")
      # term_evals = term_evals
    )

    # auto tuner kknn
    at_kknn = auto_tuner(
      tuner = tuner_,
      learner = graph_kknn,
      resampling = custom_,
      measure = measure_,
      search_space = search_space_kknn,
      terminator = trm("none")
      # term_evals = term_evals
    )

    # auto tuner gbm
    at_gbm = auto_tuner(
      tuner = tuner_,
      learner = graph_gbm,
      resampling = custom_,
      measure = measure_,
      search_space = search_space_gbm,
      terminator = trm("none")
      # term_evals = term_evals
    )

    # auto tuner rsm
    at_rsm = auto_tuner(
      tuner = tuner_,
      learner = graph_rsm,
      resampling = custom_,
      measure = measure_,
      search_space = search_space_rsm,
      terminator = trm("none")
      # term_evals = term_evals
    )

    # auto tuner rsm
    at_bart = auto_tuner(
      tuner = tuner_,
      learner = graph_bart,
      resampling = custom_,
      measure = measure_,
      search_space = search_space_bart,
      terminator = trm("none")
      # term_evals = term_evals
    )

    # auto tuner catboost
    at_catboost = auto_tuner(
      tuner = tuner_,
      learner = graph_catboost,
      resampling = custom_,
      measure = measure_,
      search_space = search_space_catboost,
      terminator = trm("none")
      # term_evals = term_evals
    )

    # auto tuner glmnet
    at_glmnet = auto_tuner(
      tuner = tuner_,
      learner = graph_glmnet,
      resampling = custom_,
      measure = measure_,
      search_space = search_space_glmnet,
      terminator = trm("none")
      # term_evals = term_evals
    )

    # auto tuner glmnet
    at_cforest = auto_tuner(
      tuner = tuner_,
      learner = graph_cforest,
      resampling = custom_,
      measure = measure_,
      search_space = search_space_cforest,
      terminator = trm("none")
      # term_evals = term_evals
    )

    # outer resampling
    customo_ = rsmp("custom")
    customo_$id = paste0("custom_", cv_inner$iters, "_", i)
    customo_$instantiate(tasks[[j]], list(cv_outer$train_set(i)), list(cv_outer$test_set(i)))

    # nested CV for one round
    design = benchmark_grid(
      tasks = tasks[[j]]$clone(),
      learners = list(at_rf, at_xgboost, at_lightgbm, at_nnet, at_earth,
                      at_kknn, at_gbm, at_rsm, at_bart, at_catboost, at_glmnet,
                      at_cforest),
      resamplings = customo_
    )
  })
  designs_cv = do.call(rbind, designs_cv_l)
})
designs = do.call(rbind, designs_parallel_l)

# Exp dir
if (interactive()) {
  dirname_ = "experiments_test"
  if (dir.exists(dirname_)) system(paste0("rm -r ", dirname_))
} else {
  dirname_ = "experiments"
  if (dir.exists(dirname_)) system(paste0("rm -r ", dirname_))
}

# Create registry
packages = c(
  "data.table", "gausscov", "paradox", "mlr3", "mlr3pipelines", "mlr3tuning",
  "mlr3misc", "future", "future.apply", "mlr3extralearners", "stats"
)
reg = makeExperimentRegistry(file.dir = dirname_,
                             seed = 1,
                             packages = packages)

# Populate registry with problems and algorithms to form the jobs
batchmark(designs, store_models = FALSE, reg = reg)

# Save registry
saveRegistry(reg = reg)

# Train if local else save as file
if (interactive()) {
  # get nondone jobs
  ids = findNotDone(reg = reg)

  # set up cluster (for local it is parallel)
  cf = makeClusterFunctionsSocket(ncpus = ncpus)
  reg$cluster.functions = cf
  saveRegistry(reg = reg)

  # test one job
  test = batchtools::testJob(1, reg = reg)
  test$prediction$test$response

  # # define resources and submit jobs
  # resources = list(ncpus = 2, memory = 8000)
  # submitJobs(ids = ids$job.id,
  #            resources = resources,
  #            reg = reg)
} else {
  # create sh file
  sh_file = sprintf("
#!/bin/bash

#PBS -N MLOHLCV
#PBS -l ncpus=%d
#PBS -l mem=4GB
#PBS -J 1-%d
#PBS -o %s/logs
#PBS -j oe

cd ${PBS_O_WORKDIR}
apptainer run image.sif padobran_run.R 0 %s
", ncpus, nrow(designs), dirname_, dirname_)
  sh_file_name = "padobran_run.sh"
  file.create(sh_file_name)
  writeLines(sh_file, sh_file_name)
}
