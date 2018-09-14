#!/usr/bin/env Rscript

library("lme4")
source(system.file("utils", "allFit.R", package="lme4"))

nofoil <- FALSE

filestem <- "models/accuracy-model-"
if (nofoil) filestem <- paste0(filestem, "nofoil-")

# models <- c("full", "fourway", "threeway")
# if (!nofoil) models <- c(models, c("twoway", "noslot", "noslot_threeway",
#                                    "noslot_twoway", "twoway_truth"))
models <- c("nobias_threeway", "nobias_twoway",
            "somebias_threeway", "somebias_twoway")

print_allfit_stats <- function(mod) {
    ss <- summary(refit_all)
    ## which optimizers successfully fit the data
    print(ss$which.OK)
    ## model estimates from the various optimizers
    print(ss$fixef)
    ## stddev of model estimates from the various optimizers
    print(apply(ss$fixef, 2, sd))
    print(max(apply(ss$fixef, 2, sd)))
    ss
}

relative_likelihood <- function(mod_a, mod_b, threshold=0.05) {
    # for comparing non-nested models
    the_args <- as.list(match.call())
    modnames <- sapply(c(the_args$mod_a, the_args$mod_b), as.character)
    aics <- c(extractAIC(mod_a)[2], extractAIC(mod_b)[2])
    hi <- which.max(aics)
    lo <- which.min(aics)
    relative_likelihood <- exp((aics[lo] - aics[hi]) / 2)
    better_fit <- ifelse(relative_likelihood < threshold, hi, lo)
    message <- ifelse(relative_likelihood < threshold, "more complex", "simpler")
    cat("better fit: ", message, " model (", modnames[better_fit], ", p=",
               round(relative_likelihood, 6), ")\n", sep="")
    invisible(relative_likelihood)
}


## load data
load("clean-accuracy-data.RData")  # acc_df
for (model in models) {
    load(paste0(filestem, model, ".RData"))
    assign(model, mod, envir=.GlobalEnv)
    cat(paste0(model, "\n"))
    print(mod@optinfo$conv$lme4$messages)
    cat("\n\n")
}

stop("YOU SHOULD BE RUNNING THIS SCRIPT INTERACTIVELY; EDIT BELOW TO TEST",
     "SPECIFIC PAIRS OF MODELS.")

## use the "allfit" versions to assess candidate models
load("models/accuracy-model-nofoil-allfit-fourway.RData")  # provides refit_all
ss <- print_allfit_stats(refit_all)
## example: 4 of the 7 optimizers converged, and among them there is good
## agreement on the coef. estimates; largest across-optimizer SD is 0.001 (a
## very small effect on the relevant scale), so we conclude that the model
## is in fact well-fit, and proceed to model comparison, followed by
## final fitting / post-hocs in another script.

## model selection
anova(model_1, model_2)
## or, if models are not fully nested (so `anova(XXX, YYY)` isn't appropriate),
## we can compute "relative likelihood" instead. This gives the probability
## that "YYY" (the one with higher AIC) minimizes information loss at least as
## well as "XXX" (the one with lower AIC).
relative_likelihood(XXX, YYY)
