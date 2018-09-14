#!/usr/bin/env Rscript
#
# Author: Dan McCloy <drmccloy@uw.edu>
# License: BSD (3-clause)

suppressPackageStartupMessages(library(afex))
library(parallel)

## LOAD DATA
cat("loading data\n")
load("clean-rt-data.RData")  # rt_df
out_dir <- "models"

## SET UP CLUSTER
cat("starting cluster\n")
log <- file.path(out_dir, "cluster-output.log")
invisible(file.remove(log))
cl <- makeForkCluster(nnodes=3, outfile=log)

## FIT REACTION TIME MODELS
form <- formula(rt ~ ldiff*space*attn*slot + (1|subj))
cat("fitting RT model (KR)\n")
proc.time()
mod <- mixed(form, data=rt_df, method="KR", check_contrasts=FALSE,
             cl=cl, control=lmerControl(optCtrl=list(maxfun=30000)))
save(mod, file=file.path(out_dir, "rt-model-final.RData"))

proc.time()
stopCluster(cl)
