#!/usr/bin/env Rscript

suppressPackageStartupMessages(library(dplyr))
suppressPackageStartupMessages(library(afex))

## load data
load(file.path("models", "accuracy-model-nobias-final.RData"))  # provides `mod`
emm_options(pbkrtest.limit=10000, lmerTest.limit=10000)
outdir <- "posthocs"
dir.create(outdir, showWarnings=FALSE)

# print(mod$anova_table)
# print(summary(mod$full_model))

stars <- function(pval) {
    ifelse(pval < 0.001, "***", ifelse(pval < 0.01, "**",
                                       ifelse(pval < 0.05, "*", "")))
}

replace_names <- function(x) {
    x <- replace(x, x=="non-spatial", "non-\nspatial")
    x <- replace(x, x=="ldiff", "listening\ndifficulty")
    x <- replace(x, x=="ctrl", "control")
}

## MAIN EFFECTS
maineffs <- c("ldiff", "attn", "space")
for (maineff in maineffs) {
    ## make emmeans objects
    contr <- switch(maineff, attn="trt.vs.ctrlk", ldiff="trt.vs.ctrlk",
                    space="revpairwise")
    posthoc <- emmeans(mod, spec=maineff, by="truth", type="response",
                       contr=contr)
    assign(maineff, posthoc, envir=.GlobalEnv)
    ## make dataframe for plotting
    name <- paste0(maineff, "_emmeans")
    as.data.frame(get(maineff)$emmeans) %>%
        rename(level=maineff, token=truth) %>%
        mutate(variable=maineff,
               level=as.character(level),
               level=replace(level, level=="non-spatial", "non-\nspatial"),
               level=replace(level, level=="ldiff", "listening\ndifficulty"),
               level=replace(level, level=="ctrl", "control")) %>%
        select(variable, level, token, everything()) ->
        this_df
    assign(name, this_df, envir=.GlobalEnv)
    ## make dataframe for post-hocs
    name <- paste0(maineff, "_posthocs")
    as.data.frame(get(maineff)$contrasts) %>%
        rename(token=truth) %>%
        mutate(variable=maineff, signif=stars(p.value)) %>%
        select(variable, contrast, token, everything()) ->
        this_df
    assign(name, this_df, envir=.GlobalEnv)
}
## aggregate
main_effects <- bind_rows(ldiff_emmeans, attn_emmeans, space_emmeans)
maineff_posthocs <- bind_rows(ldiff_posthocs, attn_posthocs, space_posthocs)

## INTERACTIONS
interacts <- list(c("ldiff", "space"), c("attn", "ldiff"), c("space", "attn"))
for (interact in interacts) {
    ## make emmeans objects
    spec <- interact[1]
    grby <- interact[2]
    contr <- switch(spec, attn="trt.vs.ctrlk", ldiff="trt.vs.ctrlk",
                    space="revpairwise")
    posthoc <- emmeans(mod, spec=spec, by=c(grby, "truth"),
                       type="response", contr=contr)
    name <- paste(interact, collapse="_")
    assign(name, posthoc, envir=.GlobalEnv)
    ## make dataframe for plotting
    name_emm <- paste0(name, "_emmeans")
    as.data.frame(get(name)$emmeans) %>%
        rename(level=spec, group=grby, token=truth) %>%
        mutate(grouping=grby, variable=spec,
               level=as.character(level),
               level=replace_names(level)) %>%
        select(grouping, group, variable, level, token, everything()) ->
        this_df
    assign(name_emm, this_df, envir=.GlobalEnv)
    ## make dataframe for post-hocs
    name_ph <- paste0(name, "_posthocs")
    as.data.frame(get(name)$contrasts) %>%
        rename(group=grby, token=truth) %>%
        mutate(grouping=grby, variable=spec, signif=stars(p.value),
               group=as.character(group),
               group=replace_names(group)) %>%
        select(grouping, group, variable, contrast, token, everything()) ->
        this_df
    assign(name_ph, this_df, envir=.GlobalEnv)
}
## aggregate
interactions <- bind_rows(ldiff_space_emmeans, attn_ldiff_emmeans,
                          space_attn_emmeans)
interact_posthocs <- bind_rows(ldiff_space_posthocs, attn_ldiff_posthocs,
                               space_attn_posthocs)

## SAVE
write.csv(main_effects, file=file.path(outdir, "main-effects-estimates.csv"),
          row.names=FALSE)
write.csv(interactions, file=file.path(outdir, "interaction-estimates.csv"),
          row.names=FALSE)
write.csv(maineff_posthocs, file=file.path(outdir, "main-effects-posthocs.csv"),
          row.names=FALSE)
write.csv(interact_posthocs, file=file.path(outdir, "interaction-posthocs.csv"),
          row.names=FALSE)

## MODEL SUMMARY
modsum <- summary(mod$full_model)$coefficients
renamer <- c(`(Intercept)`="neither:::",
             `truth_target`="target:::",
             `truth_foil`="foil:::",
             `truthneither:ldiff_ldiff`="neither:ldiff::",
             `truthtarget:ldiff_ldiff`="target:ldiff::",
             `truthfoil:ldiff_ldiff`="foil:ldiff::",
             `truthneither:attn_switch`="neither::switch:",
             `truthtarget:attn_switch`="target::switch:",
             `truthfoil:attn_switch`="foil::switch:",
             `truthneither:space_non-spatial`="neither:::non-spatial",
             `truthtarget:space_non-spatial`="target:::non-spatial",
             `truthfoil:space_non-spatial`="foil:::non-spatial",
             `truthneither:space_spatial`="neither:::spatial",
             `truthtarget:space_spatial`="target:::spatial",
             `truthfoil:space_spatial`="foil:::spatial",
             `truthneither:ldiff_ldiff:attn_switch`="neither:ldiff:switch:",
             `truthtarget:ldiff_ldiff:attn_switch`="target:ldiff:switch:",
             `truthfoil:ldiff_ldiff:attn_switch`="foil:ldiff:switch:",
             `truthneither:ldiff_ldiff:space_non-spatial`="neither:ldiff::non-spatial",
             `truthtarget:ldiff_ldiff:space_non-spatial`="target:ldiff::non-spatial",
             `truthfoil:ldiff_ldiff:space_non-spatial`="foil:ldiff::non-spatial",
             `truthneither:ldiff_ldiff:space_spatial`="neither:ldiff::spatial",
             `truthtarget:ldiff_ldiff:space_spatial`="target:ldiff::spatial",
             `truthfoil:ldiff_ldiff:space_spatial`="foil:ldiff::spatial",
             `truthneither:attn_switch:space_non-spatial`="neither::switch:non-spatial",
             `truthtarget:attn_switch:space_non-spatial`="target::switch:non-spatial",
             `truthfoil:attn_switch:space_non-spatial`="foil::switch:non-spatial",
             `truthneither:attn_switch:space_spatial`="neither::switch:spatial",
             `truthtarget:attn_switch:space_spatial`="target::switch:spatial",
             `truthfoil:attn_switch:space_spatial`="foil::switch:spatial")
col_renamer <- c(Estimate="coef", `Std. Error`="stderr", `z value`="z", `Pr(>|z|)`="p")
rownames(modsum) <- renamer[rownames(modsum)]
colnames(modsum) <- col_renamer[colnames(modsum)]

modsum %>%
    as.data.frame() %>%
    mutate(effect=rownames(.), signif=stars(p)) %>%
    tidyr::separate(effect, into=c("truth", "ldiff", "attn", "space"), sep=":") %>%
    select(truth, ldiff, attn, space, everything()) ->
    modsum
write.csv(modsum, file=file.path(outdir, "model-summary.csv"), row.names=FALSE)
