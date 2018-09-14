#!/usr/bin/env Rscript

suppressPackageStartupMessages(library(dplyr))
suppressPackageStartupMessages(library(afex))

## load data
load(file.path("models", "rt-model-final.RData"))  # provides `mod`
emm_options(pbkrtest.limit=10000, lmerTest.limit=10000)
outdir <- "posthocs"
dir.create(outdir, showWarnings=FALSE)

# print(mod$anova_table)
# print(summary(mod))

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
maineffs <- c("ldiff", "attn", "space", "slot")
for (maineff in maineffs) {
    ## make emmeans objects
    contr <- switch(maineff, attn="trt.vs.ctrlk", ldiff="trt.vs.ctrlk",
                    space="revpairwise", slot="revpairwise")
    posthoc <- emmeans(mod, spec=maineff, contr=contr)
    assign(maineff, posthoc, envir=.GlobalEnv)
    ## make dataframe for plotting
    name <- paste0(maineff, "_emmeans")
    as.data.frame(get(maineff)$emmeans) %>%
        rename(level=maineff) %>%
        mutate(variable=maineff,
               level=as.character(level),
               level=replace(level, level=="non-spatial", "non-\nspatial"),
               level=replace(level, level=="ldiff", "listening\ndifficulty"),
               level=replace(level, level=="ctrl", "control")) %>%
        select(variable, level, everything()) ->
        this_df
    assign(name, this_df, envir=.GlobalEnv)
    ## make dataframe for post-hocs
    name <- paste0(maineff, "_posthocs")
    as.data.frame(get(maineff)$contrasts) %>%
        mutate(variable=maineff, signif=stars(p.value)) %>%
        select(variable, contrast, everything()) ->
        this_df
    assign(name, this_df, envir=.GlobalEnv)
}
## aggregate
main_effects <- bind_rows(ldiff_emmeans, attn_emmeans, space_emmeans,
                          slot_emmeans)
maineff_posthocs <- bind_rows(ldiff_posthocs, attn_posthocs, space_posthocs,
                              slot_posthocs)

## TWOWAY INTERACTIONS
twoways <- list(c("ldiff", "space"), c("ldiff", "attn"), c("attn", "space"),
                  c("ldiff", "slot"), c("attn", "slot"), c("space", "slot"))
for (interact in twoways) {
    ## make emmeans objects
    spec <- interact[1]
    grby <- interact[2]
    contr <- switch(spec, attn="trt.vs.ctrlk", ldiff="trt.vs.ctrlk",
                    space="revpairwise", slot="revpairwise")
    posthoc <- emmeans(mod, spec=spec, by=grby, contr=contr)
    name <- paste(interact, collapse="_")
    assign(name, posthoc, envir=.GlobalEnv)
    ## make dataframe for plotting
    name_emm <- paste0(name, "_emmeans")
    as.data.frame(get(name)$emmeans) %>%
        rename(level=spec, group=grby) %>%
        mutate(grouping=grby, variable=spec,
               level=as.character(level),
               level=replace_names(level)) %>%
        select(grouping, group, variable, level, everything()) ->
        this_df
    assign(name_emm, this_df, envir=.GlobalEnv)
    ## make dataframe for post-hocs
    name_ph <- paste0(name, "_posthocs")
    as.data.frame(get(name)$contrasts) %>%
        rename(group=grby) %>%
        mutate(grouping=grby, variable=spec, signif=stars(p.value),
               group=as.character(group),
               group=replace_names(group)) %>%
        select(grouping, group, variable, contrast, everything()) ->
        this_df
    assign(name_ph, this_df, envir=.GlobalEnv)
}
## aggregate
twoways <- bind_rows(ldiff_space_emmeans, ldiff_attn_emmeans,
                     attn_space_emmeans, ldiff_slot_emmeans,
                     attn_slot_emmeans, space_slot_emmeans)
twoway_posthocs <- bind_rows(ldiff_space_posthocs, ldiff_attn_posthocs,
                             attn_space_posthocs, ldiff_slot_posthocs,
                             attn_slot_posthocs, space_slot_posthocs)

## THREEWAY INTERACTIONS
threeways <- list(c("ldiff", "space", "slot"), c("ldiff", "attn", "slot"),
                  c("attn", "space", "slot"), c("ldiff", "attn", "space"))
for (interact in threeways) {
    ## make emmeans objects
    spec <- interact[1]
    grby <- interact[2:3]
    grby_name <- paste(grby, collapse="_")
    contr <- switch(spec, attn="trt.vs.ctrlk", ldiff="trt.vs.ctrlk",
                    space="revpairwise", slot="revpairwise")
    posthoc <- emmeans(mod, spec=spec, by=grby, contr=contr)
    name <- paste(interact, collapse="_")
    assign(name, posthoc, envir=.GlobalEnv)
    ## make dataframe for plotting
    name_emm <- paste0(name, "_emmeans")
    as.data.frame(get(name)$emmeans) %>%
        rename(level=spec, group1=grby[1], group2=grby[2]) %>%
        mutate(grouping=grby_name, variable=spec,
               level=as.character(level),
               level=replace_names(level)) %>%
        select(grouping, group1, group2, variable, level, everything()) ->
        this_df
    assign(name_emm, this_df, envir=.GlobalEnv)
    ## make dataframe for post-hocs
    name_ph <- paste0(name, "_posthocs")
    as.data.frame(get(name)$contrasts) %>%
        rename(group1=grby[1], group2=grby[2]) %>%
        mutate(grouping=grby_name, variable=spec, signif=stars(p.value),
               group1=as.character(group1),
               group2=as.character(group2),
               group1=replace_names(group1),
               group2=replace_names(group2)) %>%
        select(grouping, group1, group2, variable, contrast, everything()) ->
        this_df
    assign(name_ph, this_df, envir=.GlobalEnv)
}
## aggregate
threeways <- bind_rows(ldiff_space_slot_emmeans, ldiff_attn_slot_emmeans,
                       attn_space_slot_emmeans, ldiff_attn_space_emmeans)
threeway_posthocs <- bind_rows(ldiff_space_slot_posthocs,
                               ldiff_attn_slot_posthocs,
                               attn_space_slot_posthocs,
                               ldiff_attn_space_posthocs)

## FOURWAY INTERACTION
fourway <- c("ldiff", "attn", "space", "slot")
posthoc <- emmeans(mod, spec="ldiff", by=c("attn", "space", "slot"),
                   contr="trt.vs.ctrlk")
as.data.frame(posthoc$emmeans) %>%
    rename(level="ldiff", group1="attn", group2="space", group3="slot") %>%
    mutate(grouping="attn_space_slot", variable="ldiff",
           level=as.character(level),
           level=replace_names(level)) %>%
    select(grouping, group1, group2, group3, variable, level, everything()) ->
    fourways
as.data.frame(posthoc$contrasts) %>%
    rename(group1="attn", group2="space", group3="slot") %>%
    mutate(grouping="attn_space_slot", variable="ldiff", signif=stars(p.value),
           group1=as.character(group1),
           group2=as.character(group2),
           group3=as.character(group3),
           group1=replace_names(group1),
           group2=replace_names(group2),
           group3=replace_names(group3)) %>%
    select(grouping, group1, group2, group3, variable, contrast, everything()) ->
    fourway_posthocs


## SAVE
write.csv(main_effects, file=file.path(outdir, "main-effect-rt-estimates.csv"),
          row.names=FALSE)
write.csv(twoways, file=file.path(outdir, "twoway-rt-estimates.csv"),
          row.names=FALSE)
write.csv(threeways, file=file.path(outdir, "threeway-rt-estimates.csv"),
          row.names=FALSE)
write.csv(fourways, file=file.path(outdir, "fourway-rt-estimates.csv"),
          row.names=FALSE)

write.csv(maineff_posthocs, file=file.path(outdir, "main-effect-rt-posthocs.csv"),
          row.names=FALSE)
write.csv(twoway_posthocs, file=file.path(outdir, "twoway-rt-posthocs.csv"),
          row.names=FALSE)
write.csv(threeway_posthocs, file=file.path(outdir, "threeway-rt-posthocs.csv"),
          row.names=FALSE)
write.csv(fourway_posthocs, file=file.path(outdir, "fourway-rt-posthocs.csv"),
          row.names=FALSE)

## MODEL SUMMARY
modsum <- summary(mod$full_model)$coefficients
renamer <- c(`(Intercept)`=":::",
             `ldiff_ldiff`="ldiff:::",
             `space_non-spatial`=":non-spatial::",
             `space_spatial`=":spatial::",
             `attn_switch`="::switch:",
             `slot_2`=":::2",
             `slot_3`=":::3",
             `slot_4`=":::4",
             `ldiff_ldiff:space_non-spatial`="ldiff:non-spatial::",
             `ldiff_ldiff:space_spatial`="ldiff:spatial::",
             `ldiff_ldiff:attn_switch`="ldiff::switch:",
             `space_non-spatial:attn_switch`=":non-spatial:switch:",
             `space_spatial:attn_switch`=":spatial:switch:",
             `ldiff_ldiff:slot_2`="ldiff:::2",
             `ldiff_ldiff:slot_3`="ldiff:::3",
             `ldiff_ldiff:slot_4`="ldiff:::4",
             `space_non-spatial:slot_2`=":non-spatial::2",
             `space_spatial:slot_2`=":spatial::2",
             `space_non-spatial:slot_3`=":non-spatial::3",
             `space_spatial:slot_3`=":spatial::3",
             `space_non-spatial:slot_4`=":non-spatial::4",
             `space_spatial:slot_4`=":spatial::4",
             `attn_switch:slot_2`="::switch:2",
             `attn_switch:slot_3`="::switch:3",
             `attn_switch:slot_4`="::switch:4",
             `ldiff_ldiff:space_non-spatial:attn_switch`="ldiff:non-spatial:switch:",
             `ldiff_ldiff:space_spatial:attn_switch`="ldiff:spatial:switch:",
             `ldiff_ldiff:space_non-spatial:slot_2`="ldiff:non-spatial::2",
             `ldiff_ldiff:space_spatial:slot_2`="ldiff:spatial::2",
             `ldiff_ldiff:space_non-spatial:slot_3`="ldiff:non-spatial::3",
             `ldiff_ldiff:space_spatial:slot_3`="ldiff:spatial::3",
             `ldiff_ldiff:space_non-spatial:slot_4`="ldiff:non-spatial::4",
             `ldiff_ldiff:space_spatial:slot_4`="ldiff:spatial::4",
             `ldiff_ldiff:attn_switch:slot_2`="ldiff::switch:2",
             `ldiff_ldiff:attn_switch:slot_3`="ldiff::switch:3",
             `ldiff_ldiff:attn_switch:slot_4`="ldiff::switch:4",
             `space_non-spatial:attn_switch:slot_2`=":non-spatial:switch:2",
             `space_spatial:attn_switch:slot_2`=":spatial:switch:2",
             `space_non-spatial:attn_switch:slot_3`=":non-spatial:switch:3",
             `space_spatial:attn_switch:slot_3`=":spatial:switch:3",
             `space_non-spatial:attn_switch:slot_4`=":non-spatial:switch:4",
             `space_spatial:attn_switch:slot_4`=":spatial:switch:4",
             `ldiff_ldiff:space_non-spatial:attn_switch:slot_2`="ldiff:non-spatial:switch:2",
             `ldiff_ldiff:space_spatial:attn_switch:slot_2`="ldiff:spatial:switch:2",
             `ldiff_ldiff:space_non-spatial:attn_switch:slot_3`="ldiff:non-spatial:switch:3",
             `ldiff_ldiff:space_spatial:attn_switch:slot_3`="ldiff:spatial:switch:3",
             `ldiff_ldiff:space_non-spatial:attn_switch:slot_4`="ldiff:non-spatial:switch:4",
             `ldiff_ldiff:space_spatial:attn_switch:slot_4`="ldiff:spatial:switch:4")
col_renamer <- c(Estimate="coef", `Std. Error`="stderr", df="df",
                 `t value`="t", `Pr(>|t|)`="p")
rownames(modsum) <- renamer[rownames(modsum)]
colnames(modsum) <- col_renamer[colnames(modsum)]

modsum %>%
    as.data.frame() %>%
    mutate(effect=rownames(.), signif=stars(p)) %>%
    tidyr::separate(effect, into=c("ldiff", "space", "attn", "slot"), sep=":") %>%
    select(ldiff, attn, space, slot, everything()) ->
    modsum
write.csv(modsum, file=file.path(outdir, "rt-model-summary.csv"),
          row.names=FALSE)
