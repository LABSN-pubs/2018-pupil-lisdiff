#!/usr/bin/env Rscript
#
# Author: Dan McCloy <drmccloy@uw.edu>
# License: BSD (3-clause)
modelspec <- commandArgs(trailingOnly=TRUE)
if (length(modelspec) != 1) {
    stop("script requires exactly 1 command line argument specifying model",
         "complexity ('full', 'fourway', 'threeway', 'noslot', etc)")
}
modelspecs <- list(full=formula(press ~ truth*lisdiff*space*attn*slot + (1|subj)),
                   fourway=formula(press ~ (truth+lisdiff+space+attn+slot)^4 + (1|subj)),
                   threeway=formula(press ~ (truth+lisdiff+space+attn+slot)^3 + (1|subj)),
                   twoway=formula(press ~ (truth+lisdiff+space+attn+slot)^2 + (1|subj)),
                   twoway_truth=formula(press ~ truth*(lisdiff+space+attn+slot)^2 + (1|subj)),
                   noslot_fourway=formula(press ~ truth*lisdiff*space*attn + (1|subj)),
                   noslot_threeway=formula(press ~ (truth+lisdiff+space+attn)^3 + (1|subj)),
                   noslot_twoway=formula(press ~ (truth+lisdiff+space+attn)^2 + (1|subj)),
                   nobias_threeway=formula(press ~ truth + truth:(ldiff + attn + space)^3 + (1|subj)),
                   nobias_twoway=formula(press ~ truth + truth:(ldiff + attn + space)^2 + (1|subj)),
                   somebias_threeway=formula(press ~ truth + ldiff + attn + space + truth:(ldiff + attn + space)^3 + (1|subj)),
                   somebias_twoway=formula(press ~ truth + ldiff + attn + space + truth:(ldiff + attn + space)^2 + (1|subj)))
form <- modelspecs[[modelspec]]
library(lme4)

get_pars <- function(model) {
    require("lme4")
    ## GLMM diagnostics require both random and fixed parameters
    if (isLMM(model)) pars <- getME(model, "theta")
    else              pars <- getME(model, c("theta", "fixef"))
    pars
}

check_cholesky_diag <- function(model) {
    require("lme4")
    diag_vals <- getME(model, "theta")[getME(model, "lower") == 0]
    if (any(diag_vals < 1e-6)) {
        stop("small values in cholesky diagonal, may be singular")
    }
}

melt_hess <- function(hess, source=NULL) {
    require("reshape2")
    longhess <- melt(hess)
    names(longhess) <- c("x", "y", "h")
    if (!is.null(source)) longhess$source <- source
    longhess
}

recompute_grad_hess <- function(model) {
    # recompute gradient and Hessian with Richardson extrapolation
    require("lme4")
    devfun <- update(model, devFunOnly=TRUE)
    pars <- get_pars(model)
    if (require("numDeriv")) {
        hess <- hessian(devfun, unlist(pars))
        grad <- grad(devfun, unlist(pars))
        scaled_grad <- solve(chol(hess), grad)
        result <- list(gradient=grad, Hessian=hess)
    }
    result
}

plot_hess_grad <- function(old, new) {
    require("ggplot2")
    require("egg")
    hess_args <- list(theme_bw(), scale_fill_gradient2())
    old_grad <- old$gradient
    old_hess <- melt_hess(old$Hessian, source="finite diffs")
    new_grad <- new$gradient
    new_hess <- melt_hess(new$Hessian, source="Richardson")
    old_h <- qplot(x, y, fill=h, data=old_hess, geom="tile") + hess_args
    new_h <- qplot(x, y, fill=h, data=new_hess, geom="tile") + hess_args
    old_g <- qplot(seq_along(old_grad), old_grad, geom="line") + theme_bw()
    new_g <- qplot(seq_along(new_grad), new_grad, geom="line") + theme_bw()
    gtab <- ggarrange(old_g, new_g, old_h, new_h, nrow=2, heights=c(1, 6),
                      labels=c("old gradient", "new gradient", "old Hessian",
                               "new Hessian"))
}


## LOAD DATA
cat("loading data\n")
load(file.path("processed-data", "clean-accuracy-data.RData"))  # acc_df
out_dir <- "models"

## FIT ACCURACY MODEL
proc.time()
cat("fitting accuracy model\n")
mod <- glmer(form, data=acc_df, family=binomial(link="probit"),
             control=glmerControl(optCtrl=list(maxfun=30000)))
cat("model fitted\n")
proc.time()
#spec <- gsub(" ", "", as.character(form)[3])
fname <- paste0("accuracy-model-", modelspec, ".RData")
save(mod, file=file.path(out_dir, fname))

## DIAGNOSTICS
if (!mod@optinfo$conv$opt) {
    ## check for very small diagonal entries in Cholesky decomposition
    check_cholesky_diag(mod)

    ## check quick-and-dirty Hessian estimate against Richardson method
    old <- mod@optinfo$derivs
    new <- recompute_grad_hess(mod)
    cairo_pdf(file.path(out_dir, paste0(modelspec, ".pdf")), width=10,
              height=8)
    plot_hess_grad(old, new)
    dev.off()
    stopifnot(all.equal(old$Hessian, new$Hessian, tolerance=1e-3))

    ## refit with starting params == ending params
    refit <- update(mod, start=get_pars(mod))
    if (refit@optinfo$conv$opt) stop("refit converged")
    else {
        ## try all optimizers
        source(system.file("utils", "allFit.R", package="lme4"))
        refit_all <- allFit(mod)
        ss <- summary(refit_all)
        print(ss$which.OK)
        fname <- paste0("accuracy-model-allfit-", modelspec, ".RData")
        save(refit_all, file=file.path(out_dir, fname))
    }
}
