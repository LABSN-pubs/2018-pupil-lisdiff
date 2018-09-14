#!/usr/bin/env Rscript
#
# Author: Dan McCloy <drmccloy@uw.edu>
# License: BSD (3-clause)

## ## ## ## ## ## ## ## ## ## ## ## ## ##
##  FUNCTIONS TO SET FACTOR CONTRASTS  ##
## ## ## ## ## ## ## ## ## ## ## ## ## ##

## treatment contrasts
txContrast <- function(x, ...) {
    x <- factor(x, ...)
    contrasts(x) <- contr.treatment
    colnames(contrasts(x)) <- paste0("_", levels(x)[-1])
    x
}

## sum-to-one (deviation contrasts)
devContrast <- function(x, ...) {
    x <- factor(x, ...)
    contrasts(x) <- contr.sum
    contrasts(x) <- contrasts(x) / 2
    colnames(contrasts(x)) <- paste0("_", levels(x)[-length(levels(x))])
    x
}

## ## ## ## ## ##
##  LOAD DATA  ##
## ## ## ## ## ##

## file paths
data_dir <- "processed-data"
out_dir <- "models"
dir.create(out_dir, showWarnings=FALSE)
log <- file.path(out_dir, "cluster-log.txt")

## data types
col_classes <- c(subj="integer", lisdiff="logical", group="character",
                 trial="integer", trial_id=NULL, lr="character",
                 mf="character", attn="character", s_cond="character",
                 presses="character", slot="integer", slot_code="character",
                 slot_onset="numeric", rt="numeric", extra_rts=NULL,
                 h="integer", m="integer", f="integer", c="integer",
                 ff="logical", fo="logical", fd="integer", cf="logical",
                 co="logical")
## PARTIAL KEY
## h = hit target   |  f = total F.A.  |  ff = resp foil      fo = resp other
## m = miss target  |  c = total C.R.  |  cf = no resp foil   co = no resp other
##
## note that non-responses to non-targ-non-foil slots are not counted as
## correct rejections by some authors; here the modeling effectively forces
## them to be counted as such (as it models press/nopress vs. targ/foil/neither)

## load
slot_df <- read.delim(file.path(data_dir, "behavioral-data-by-slot.tsv"),
                      sep="\t", colClasses=col_classes)

## convert some logicals to integers
#columns <- c("ff", "fo", "cf", "co")
#slot_df[columns] <- data.matrix(slot_df[columns])

## set up SDT-style mixed model predictors
slot_df$press <- !is.na(slot_df$rt)
slot_df$ldiff <- devContrast(ifelse(slot_df$lisdiff, "ldiff", "ctrl"),
                             levels=c("ldiff", "ctrl"))
slot_df$attn <- devContrast(slot_df$attn, levels=c("switch", "maintain"))
slot_df$slot <- txContrast(slot_df$slot)
slot_df$space <- txContrast(slot_df$s_cond,
                            levels=c("mixed", "non-spatial", "spatial"))
slot_df$truth <- ifelse(slot_df$slot_code == "t", "target",
                        ifelse(slot_df$slot_code == "f", "foil", "neither"))
slot_df$truth <- txContrast(slot_df$truth, levels=c("neither", "target", "foil"))

## set up reaction time dataframe (only keep correct presses)
rt_df <- slot_df[slot_df$h > 0 & !is.na(slot_df$rt),
                 c("subj", "rt", "ldiff", "attn", "space", "slot")]

# ditch unnecessary columns
acc_df <- slot_df[c("subj", "press", "truth", "ldiff", "attn", "space", "slot")]

save(acc_df, file=file.path(data_dir, "clean-accuracy-data.RData"))
save(rt_df, file=file.path(data_dir, "clean-rt-data.RData"))
