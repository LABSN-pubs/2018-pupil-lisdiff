#!/usr/bin/env Rscript
#
# Author: Dan McCloy <drmccloy@uw.edu>
# License: BSD (3-clause)

library(afex)
library(parallel)

method <- "LRT"

## LOAD DATA
cat("loading data\n")
load("clean-accuracy-data.RData")  # acc_df
out_dir <- "models"

## SET UP CLUSTER
cat("starting cluster\n")
log <- file.path(out_dir, "cluster-output.log")
invisible(file.remove(log))
cl <- makeForkCluster(nnodes=6, outfile=log)

## FIT ACCURACY MODELS
proc.time()
form <- formula(press ~ truth + truth:(ldiff + attn + space)^2 + (1|subj))
if (method == "LRT") {
    cat("fitting accuracy model (LRT)\n")
    mod <- mixed(form, data=acc_df, family=binomial(link="probit"),
                 method="LRT", check_contrasts=FALSE, cl=cl,
                 control=glmerControl(optCtrl=list(maxfun=30000)))
    save(mod, file=file.path(out_dir, "accuracy-model-nobias-final.RData"))
} else if (method == "PB") {
    cat("fitting accuracy model (PB)\n")
    mod <- mixed(form, data=acc_df, family=binomial(link="probit"),
                 method="PB", check_contrasts=FALSE,
                 args_test=list(cl=cl, nsim=1000),
                 cl=cl, control=glmerControl(optCtrl=list(maxfun=30000)))
    fname <- file.path(out_dir, "accuracy-model-nobias-final-bootstrap.RData")
    save(mod, file=fname)
} else {
    stop("unknown method")
}
proc.time()
stopCluster(cl)
