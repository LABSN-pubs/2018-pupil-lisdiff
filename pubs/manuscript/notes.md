<!-- do RT as subtractions (slot 1+3 - (2 + 4)) -->
<!-- check dprime calc, why so low? -->
<!-- group diff in RT when collapsed across all slots -->
<!-- slot 3 dprime x attn x group interaction? -->

<!-- pupil AUC (switch - maintain): check for correlation against {RT, dprime, dprime by slot, spatial/nonspatial, etc} -->

<!-- speed accuracy tradeoff (dprime vs RT, esp. for slot 1, but also across all slots?) -->

<!--**TODO**: check `clean-behavioral-data.py` regarding whether slots with no “O” and no press were counted as correct rejections.  Current manuscript suggests not, but can't recall what was actually done.-->

**TODO:** move stars in accuracy graph, spatial maineff to be atop the
nonspat. / mixed instead of centered between.

# training
condition       #      # correct to pass   % correct to pass
mf lr M			5      4                   80
mf lr S			5      4                   80
mf lr M/S	   10      7                   70
mf LL M/S		5      0                    0
MM lr M/S		5      0                    0
any			   16      8                   50
