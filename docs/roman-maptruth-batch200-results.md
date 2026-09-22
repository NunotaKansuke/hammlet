# Roman / Hammlet truth-centered map-frame batch200 results

Static result page for the 200-event Roman/Hammlet sample. Each event figure
shows the best truth-centered FFT seed light curve on the left and its
corresponding \(\Delta\chi^2\) map on the right. The data points are
drawn as dark, enlarged markers in front of the model curve.

## Run note

- Event universe: `9910000`–`9910099` and `9920000`–`9920099` (200 events)
- Truth-centered FFT + LM completed: 199 events
- Separate validation figure: `9920099` (not included in the 199-event batch)
- Data points per event: `23,104`
- FFT map evaluation: `M=128`, `n_alpha=540`, radial order `1`
- Parallelism: `16 workers × 1 thread`
- Existing packed atlas only: `33,903` readable maps out of `33,993`
- No lens maps were regenerated and the atlas remained read-only

The FFT geometry grid is centered on the effective map-frame trajectory
computed from the injected truth parameters. The physical source trajectory is
not changed; the map-frame translation is applied when evaluating the legacy
AdaMGrid atlas. This is a truth-centered diagnostic run, not a blind recovery
benchmark.

## FFT and LM accounting

The 199-event batch has:

- FFT reports: `199/199`
- LM result records: `199/199`
- worker errors: `0`
- non-negative LM \(\Delta\chi^2\) improvements: `199/199`
- LM convergence flags: `3/199`

The low convergence-flag count is expected from the deliberately short
`max_nfev=25` run: the other events produced valid optimized records but
stopped at the evaluation cap.

## q/s overview

The q/s plot uses the 175 events with independently computed PSPL--truth
\(\Delta\chi^2 > 100\). It is generated from the same truth-centered
FFT seed and coordinate-corrected LM outputs.

![Truth-centered Roman q/s recovery](../assets/roman-lm-q-s-maptruth-dchi2gt100.png)

Exact selection and plotting metadata:
[sidecar JSON](../assets/roman-lm-q-s-maptruth-dchi2gt100.json).

## Figures

The event figures below are generated from the truth-centered map-frame FFT
outputs. The left panel uses the recovered FFT seed and the right panel shows
the adaptive FFT \(\Delta\chi^2\) map.

### Events 9910000–9910009

#### `9910000`

![Roman truth-centered event 9910000](../assets/roman-maptruth-batch200/9910000.png)

#### `9910001`

![Roman truth-centered event 9910001](../assets/roman-maptruth-batch200/9910001.png)

#### `9910002`

![Roman truth-centered event 9910002](../assets/roman-maptruth-batch200/9910002.png)

#### `9910003`

![Roman truth-centered event 9910003](../assets/roman-maptruth-batch200/9910003.png)

#### `9910004`

![Roman truth-centered event 9910004](../assets/roman-maptruth-batch200/9910004.png)

#### `9910005`

![Roman truth-centered event 9910005](../assets/roman-maptruth-batch200/9910005.png)

#### `9910006`

![Roman truth-centered event 9910006](../assets/roman-maptruth-batch200/9910006.png)

#### `9910007`

![Roman truth-centered event 9910007](../assets/roman-maptruth-batch200/9910007.png)

#### `9910008`

![Roman truth-centered event 9910008](../assets/roman-maptruth-batch200/9910008.png)

#### `9910009`

![Roman truth-centered event 9910009](../assets/roman-maptruth-batch200/9910009.png)

### Events 9910010–9910019

#### `9910010`

![Roman truth-centered event 9910010](../assets/roman-maptruth-batch200/9910010.png)

#### `9910011`

![Roman truth-centered event 9910011](../assets/roman-maptruth-batch200/9910011.png)

#### `9910012`

![Roman truth-centered event 9910012](../assets/roman-maptruth-batch200/9910012.png)

#### `9910013`

![Roman truth-centered event 9910013](../assets/roman-maptruth-batch200/9910013.png)

#### `9910014`

![Roman truth-centered event 9910014](../assets/roman-maptruth-batch200/9910014.png)

#### `9910015`

![Roman truth-centered event 9910015](../assets/roman-maptruth-batch200/9910015.png)

#### `9910016`

![Roman truth-centered event 9910016](../assets/roman-maptruth-batch200/9910016.png)

#### `9910017`

![Roman truth-centered event 9910017](../assets/roman-maptruth-batch200/9910017.png)

#### `9910018`

![Roman truth-centered event 9910018](../assets/roman-maptruth-batch200/9910018.png)

#### `9910019`

![Roman truth-centered event 9910019](../assets/roman-maptruth-batch200/9910019.png)

### Events 9910020–9910029

#### `9910020`

![Roman truth-centered event 9910020](../assets/roman-maptruth-batch200/9910020.png)

#### `9910021`

![Roman truth-centered event 9910021](../assets/roman-maptruth-batch200/9910021.png)

#### `9910022`

![Roman truth-centered event 9910022](../assets/roman-maptruth-batch200/9910022.png)

#### `9910023`

![Roman truth-centered event 9910023](../assets/roman-maptruth-batch200/9910023.png)

#### `9910024`

![Roman truth-centered event 9910024](../assets/roman-maptruth-batch200/9910024.png)

#### `9910025`

![Roman truth-centered event 9910025](../assets/roman-maptruth-batch200/9910025.png)

#### `9910026`

![Roman truth-centered event 9910026](../assets/roman-maptruth-batch200/9910026.png)

#### `9910027`

![Roman truth-centered event 9910027](../assets/roman-maptruth-batch200/9910027.png)

#### `9910028`

![Roman truth-centered event 9910028](../assets/roman-maptruth-batch200/9910028.png)

#### `9910029`

![Roman truth-centered event 9910029](../assets/roman-maptruth-batch200/9910029.png)

### Events 9910030–9910039

#### `9910030`

![Roman truth-centered event 9910030](../assets/roman-maptruth-batch200/9910030.png)

#### `9910031`

![Roman truth-centered event 9910031](../assets/roman-maptruth-batch200/9910031.png)

#### `9910032`

![Roman truth-centered event 9910032](../assets/roman-maptruth-batch200/9910032.png)

#### `9910033`

![Roman truth-centered event 9910033](../assets/roman-maptruth-batch200/9910033.png)

#### `9910034`

![Roman truth-centered event 9910034](../assets/roman-maptruth-batch200/9910034.png)

#### `9910035`

![Roman truth-centered event 9910035](../assets/roman-maptruth-batch200/9910035.png)

#### `9910036`

![Roman truth-centered event 9910036](../assets/roman-maptruth-batch200/9910036.png)

#### `9910037`

![Roman truth-centered event 9910037](../assets/roman-maptruth-batch200/9910037.png)

#### `9910038`

![Roman truth-centered event 9910038](../assets/roman-maptruth-batch200/9910038.png)

#### `9910039`

![Roman truth-centered event 9910039](../assets/roman-maptruth-batch200/9910039.png)

### Events 9910040–9910049

#### `9910040`

![Roman truth-centered event 9910040](../assets/roman-maptruth-batch200/9910040.png)

#### `9910041`

![Roman truth-centered event 9910041](../assets/roman-maptruth-batch200/9910041.png)

#### `9910042`

![Roman truth-centered event 9910042](../assets/roman-maptruth-batch200/9910042.png)

#### `9910043`

![Roman truth-centered event 9910043](../assets/roman-maptruth-batch200/9910043.png)

#### `9910044`

![Roman truth-centered event 9910044](../assets/roman-maptruth-batch200/9910044.png)

#### `9910045`

![Roman truth-centered event 9910045](../assets/roman-maptruth-batch200/9910045.png)

#### `9910046`

![Roman truth-centered event 9910046](../assets/roman-maptruth-batch200/9910046.png)

#### `9910047`

![Roman truth-centered event 9910047](../assets/roman-maptruth-batch200/9910047.png)

#### `9910048`

![Roman truth-centered event 9910048](../assets/roman-maptruth-batch200/9910048.png)

#### `9910049`

![Roman truth-centered event 9910049](../assets/roman-maptruth-batch200/9910049.png)

### Events 9910050–9910059

#### `9910050`

![Roman truth-centered event 9910050](../assets/roman-maptruth-batch200/9910050.png)

#### `9910051`

![Roman truth-centered event 9910051](../assets/roman-maptruth-batch200/9910051.png)

#### `9910052`

![Roman truth-centered event 9910052](../assets/roman-maptruth-batch200/9910052.png)

#### `9910053`

![Roman truth-centered event 9910053](../assets/roman-maptruth-batch200/9910053.png)

#### `9910054`

![Roman truth-centered event 9910054](../assets/roman-maptruth-batch200/9910054.png)

#### `9910055`

![Roman truth-centered event 9910055](../assets/roman-maptruth-batch200/9910055.png)

#### `9910056`

![Roman truth-centered event 9910056](../assets/roman-maptruth-batch200/9910056.png)

#### `9910057`

![Roman truth-centered event 9910057](../assets/roman-maptruth-batch200/9910057.png)

#### `9910058`

![Roman truth-centered event 9910058](../assets/roman-maptruth-batch200/9910058.png)

#### `9910059`

![Roman truth-centered event 9910059](../assets/roman-maptruth-batch200/9910059.png)

### Events 9910060–9910069

#### `9910060`

![Roman truth-centered event 9910060](../assets/roman-maptruth-batch200/9910060.png)

#### `9910061`

![Roman truth-centered event 9910061](../assets/roman-maptruth-batch200/9910061.png)

#### `9910062`

![Roman truth-centered event 9910062](../assets/roman-maptruth-batch200/9910062.png)

#### `9910063`

![Roman truth-centered event 9910063](../assets/roman-maptruth-batch200/9910063.png)

#### `9910064`

![Roman truth-centered event 9910064](../assets/roman-maptruth-batch200/9910064.png)

#### `9910065`

![Roman truth-centered event 9910065](../assets/roman-maptruth-batch200/9910065.png)

#### `9910066`

![Roman truth-centered event 9910066](../assets/roman-maptruth-batch200/9910066.png)

#### `9910067`

![Roman truth-centered event 9910067](../assets/roman-maptruth-batch200/9910067.png)

#### `9910068`

![Roman truth-centered event 9910068](../assets/roman-maptruth-batch200/9910068.png)

#### `9910069`

![Roman truth-centered event 9910069](../assets/roman-maptruth-batch200/9910069.png)

### Events 9910070–9910079

#### `9910070`

![Roman truth-centered event 9910070](../assets/roman-maptruth-batch200/9910070.png)

#### `9910071`

![Roman truth-centered event 9910071](../assets/roman-maptruth-batch200/9910071.png)

#### `9910072`

![Roman truth-centered event 9910072](../assets/roman-maptruth-batch200/9910072.png)

#### `9910073`

![Roman truth-centered event 9910073](../assets/roman-maptruth-batch200/9910073.png)

#### `9910074`

![Roman truth-centered event 9910074](../assets/roman-maptruth-batch200/9910074.png)

#### `9910075`

![Roman truth-centered event 9910075](../assets/roman-maptruth-batch200/9910075.png)

#### `9910076`

![Roman truth-centered event 9910076](../assets/roman-maptruth-batch200/9910076.png)

#### `9910077`

![Roman truth-centered event 9910077](../assets/roman-maptruth-batch200/9910077.png)

#### `9910078`

![Roman truth-centered event 9910078](../assets/roman-maptruth-batch200/9910078.png)

#### `9910079`

![Roman truth-centered event 9910079](../assets/roman-maptruth-batch200/9910079.png)

### Events 9910080–9910089

#### `9910080`

![Roman truth-centered event 9910080](../assets/roman-maptruth-batch200/9910080.png)

#### `9910081`

![Roman truth-centered event 9910081](../assets/roman-maptruth-batch200/9910081.png)

#### `9910082`

![Roman truth-centered event 9910082](../assets/roman-maptruth-batch200/9910082.png)

#### `9910083`

![Roman truth-centered event 9910083](../assets/roman-maptruth-batch200/9910083.png)

#### `9910084`

![Roman truth-centered event 9910084](../assets/roman-maptruth-batch200/9910084.png)

#### `9910085`

![Roman truth-centered event 9910085](../assets/roman-maptruth-batch200/9910085.png)

#### `9910086`

![Roman truth-centered event 9910086](../assets/roman-maptruth-batch200/9910086.png)

#### `9910087`

![Roman truth-centered event 9910087](../assets/roman-maptruth-batch200/9910087.png)

#### `9910088`

![Roman truth-centered event 9910088](../assets/roman-maptruth-batch200/9910088.png)

#### `9910089`

![Roman truth-centered event 9910089](../assets/roman-maptruth-batch200/9910089.png)

### Events 9910090–9910099

#### `9910090`

![Roman truth-centered event 9910090](../assets/roman-maptruth-batch200/9910090.png)

#### `9910091`

![Roman truth-centered event 9910091](../assets/roman-maptruth-batch200/9910091.png)

#### `9910092`

![Roman truth-centered event 9910092](../assets/roman-maptruth-batch200/9910092.png)

#### `9910093`

![Roman truth-centered event 9910093](../assets/roman-maptruth-batch200/9910093.png)

#### `9910094`

![Roman truth-centered event 9910094](../assets/roman-maptruth-batch200/9910094.png)

#### `9910095`

![Roman truth-centered event 9910095](../assets/roman-maptruth-batch200/9910095.png)

#### `9910096`

![Roman truth-centered event 9910096](../assets/roman-maptruth-batch200/9910096.png)

#### `9910097`

![Roman truth-centered event 9910097](../assets/roman-maptruth-batch200/9910097.png)

#### `9910098`

![Roman truth-centered event 9910098](../assets/roman-maptruth-batch200/9910098.png)

#### `9910099`

![Roman truth-centered event 9910099](../assets/roman-maptruth-batch200/9910099.png)

### Events 9920000–9920009

#### `9920000`

![Roman truth-centered event 9920000](../assets/roman-maptruth-batch200/9920000.png)

#### `9920001`

![Roman truth-centered event 9920001](../assets/roman-maptruth-batch200/9920001.png)

#### `9920002`

![Roman truth-centered event 9920002](../assets/roman-maptruth-batch200/9920002.png)

#### `9920003`

![Roman truth-centered event 9920003](../assets/roman-maptruth-batch200/9920003.png)

#### `9920004`

![Roman truth-centered event 9920004](../assets/roman-maptruth-batch200/9920004.png)

#### `9920005`

![Roman truth-centered event 9920005](../assets/roman-maptruth-batch200/9920005.png)

#### `9920006`

![Roman truth-centered event 9920006](../assets/roman-maptruth-batch200/9920006.png)

#### `9920007`

![Roman truth-centered event 9920007](../assets/roman-maptruth-batch200/9920007.png)

#### `9920008`

![Roman truth-centered event 9920008](../assets/roman-maptruth-batch200/9920008.png)

#### `9920009`

![Roman truth-centered event 9920009](../assets/roman-maptruth-batch200/9920009.png)

### Events 9920010–9920019

#### `9920010`

![Roman truth-centered event 9920010](../assets/roman-maptruth-batch200/9920010.png)

#### `9920011`

![Roman truth-centered event 9920011](../assets/roman-maptruth-batch200/9920011.png)

#### `9920012`

![Roman truth-centered event 9920012](../assets/roman-maptruth-batch200/9920012.png)

#### `9920013`

![Roman truth-centered event 9920013](../assets/roman-maptruth-batch200/9920013.png)

#### `9920014`

![Roman truth-centered event 9920014](../assets/roman-maptruth-batch200/9920014.png)

#### `9920015`

![Roman truth-centered event 9920015](../assets/roman-maptruth-batch200/9920015.png)

#### `9920016`

![Roman truth-centered event 9920016](../assets/roman-maptruth-batch200/9920016.png)

#### `9920017`

![Roman truth-centered event 9920017](../assets/roman-maptruth-batch200/9920017.png)

#### `9920018`

![Roman truth-centered event 9920018](../assets/roman-maptruth-batch200/9920018.png)

#### `9920019`

![Roman truth-centered event 9920019](../assets/roman-maptruth-batch200/9920019.png)

### Events 9920020–9920029

#### `9920020`

![Roman truth-centered event 9920020](../assets/roman-maptruth-batch200/9920020.png)

#### `9920021`

![Roman truth-centered event 9920021](../assets/roman-maptruth-batch200/9920021.png)

#### `9920022`

![Roman truth-centered event 9920022](../assets/roman-maptruth-batch200/9920022.png)

#### `9920023`

![Roman truth-centered event 9920023](../assets/roman-maptruth-batch200/9920023.png)

#### `9920024`

![Roman truth-centered event 9920024](../assets/roman-maptruth-batch200/9920024.png)

#### `9920025`

![Roman truth-centered event 9920025](../assets/roman-maptruth-batch200/9920025.png)

#### `9920026`

![Roman truth-centered event 9920026](../assets/roman-maptruth-batch200/9920026.png)

#### `9920027`

![Roman truth-centered event 9920027](../assets/roman-maptruth-batch200/9920027.png)

#### `9920028`

![Roman truth-centered event 9920028](../assets/roman-maptruth-batch200/9920028.png)

#### `9920029`

![Roman truth-centered event 9920029](../assets/roman-maptruth-batch200/9920029.png)

### Events 9920030–9920039

#### `9920030`

![Roman truth-centered event 9920030](../assets/roman-maptruth-batch200/9920030.png)

#### `9920031`

![Roman truth-centered event 9920031](../assets/roman-maptruth-batch200/9920031.png)

#### `9920032`

![Roman truth-centered event 9920032](../assets/roman-maptruth-batch200/9920032.png)

#### `9920033`

![Roman truth-centered event 9920033](../assets/roman-maptruth-batch200/9920033.png)

#### `9920034`

![Roman truth-centered event 9920034](../assets/roman-maptruth-batch200/9920034.png)

#### `9920035`

![Roman truth-centered event 9920035](../assets/roman-maptruth-batch200/9920035.png)

#### `9920036`

![Roman truth-centered event 9920036](../assets/roman-maptruth-batch200/9920036.png)

#### `9920037`

![Roman truth-centered event 9920037](../assets/roman-maptruth-batch200/9920037.png)

#### `9920038`

![Roman truth-centered event 9920038](../assets/roman-maptruth-batch200/9920038.png)

#### `9920039`

![Roman truth-centered event 9920039](../assets/roman-maptruth-batch200/9920039.png)

### Events 9920040–9920049

#### `9920040`

![Roman truth-centered event 9920040](../assets/roman-maptruth-batch200/9920040.png)

#### `9920041`

![Roman truth-centered event 9920041](../assets/roman-maptruth-batch200/9920041.png)

#### `9920042`

![Roman truth-centered event 9920042](../assets/roman-maptruth-batch200/9920042.png)

#### `9920043`

![Roman truth-centered event 9920043](../assets/roman-maptruth-batch200/9920043.png)

#### `9920044`

![Roman truth-centered event 9920044](../assets/roman-maptruth-batch200/9920044.png)

#### `9920045`

![Roman truth-centered event 9920045](../assets/roman-maptruth-batch200/9920045.png)

#### `9920046`

![Roman truth-centered event 9920046](../assets/roman-maptruth-batch200/9920046.png)

#### `9920047`

![Roman truth-centered event 9920047](../assets/roman-maptruth-batch200/9920047.png)

#### `9920048`

![Roman truth-centered event 9920048](../assets/roman-maptruth-batch200/9920048.png)

#### `9920049`

![Roman truth-centered event 9920049](../assets/roman-maptruth-batch200/9920049.png)

### Events 9920050–9920059

#### `9920050`

![Roman truth-centered event 9920050](../assets/roman-maptruth-batch200/9920050.png)

#### `9920051`

![Roman truth-centered event 9920051](../assets/roman-maptruth-batch200/9920051.png)

#### `9920052`

![Roman truth-centered event 9920052](../assets/roman-maptruth-batch200/9920052.png)

#### `9920053`

![Roman truth-centered event 9920053](../assets/roman-maptruth-batch200/9920053.png)

#### `9920054`

![Roman truth-centered event 9920054](../assets/roman-maptruth-batch200/9920054.png)

#### `9920055`

![Roman truth-centered event 9920055](../assets/roman-maptruth-batch200/9920055.png)

#### `9920056`

![Roman truth-centered event 9920056](../assets/roman-maptruth-batch200/9920056.png)

#### `9920057`

![Roman truth-centered event 9920057](../assets/roman-maptruth-batch200/9920057.png)

#### `9920058`

![Roman truth-centered event 9920058](../assets/roman-maptruth-batch200/9920058.png)

#### `9920059`

![Roman truth-centered event 9920059](../assets/roman-maptruth-batch200/9920059.png)

### Events 9920060–9920069

#### `9920060`

![Roman truth-centered event 9920060](../assets/roman-maptruth-batch200/9920060.png)

#### `9920061`

![Roman truth-centered event 9920061](../assets/roman-maptruth-batch200/9920061.png)

#### `9920062`

![Roman truth-centered event 9920062](../assets/roman-maptruth-batch200/9920062.png)

#### `9920063`

![Roman truth-centered event 9920063](../assets/roman-maptruth-batch200/9920063.png)

#### `9920064`

![Roman truth-centered event 9920064](../assets/roman-maptruth-batch200/9920064.png)

#### `9920065`

![Roman truth-centered event 9920065](../assets/roman-maptruth-batch200/9920065.png)

#### `9920066`

![Roman truth-centered event 9920066](../assets/roman-maptruth-batch200/9920066.png)

#### `9920067`

![Roman truth-centered event 9920067](../assets/roman-maptruth-batch200/9920067.png)

#### `9920068`

![Roman truth-centered event 9920068](../assets/roman-maptruth-batch200/9920068.png)

#### `9920069`

![Roman truth-centered event 9920069](../assets/roman-maptruth-batch200/9920069.png)

### Events 9920070–9920079

#### `9920070`

![Roman truth-centered event 9920070](../assets/roman-maptruth-batch200/9920070.png)

#### `9920071`

![Roman truth-centered event 9920071](../assets/roman-maptruth-batch200/9920071.png)

#### `9920072`

![Roman truth-centered event 9920072](../assets/roman-maptruth-batch200/9920072.png)

#### `9920073`

![Roman truth-centered event 9920073](../assets/roman-maptruth-batch200/9920073.png)

#### `9920074`

![Roman truth-centered event 9920074](../assets/roman-maptruth-batch200/9920074.png)

#### `9920075`

![Roman truth-centered event 9920075](../assets/roman-maptruth-batch200/9920075.png)

#### `9920076`

![Roman truth-centered event 9920076](../assets/roman-maptruth-batch200/9920076.png)

#### `9920077`

![Roman truth-centered event 9920077](../assets/roman-maptruth-batch200/9920077.png)

#### `9920078`

![Roman truth-centered event 9920078](../assets/roman-maptruth-batch200/9920078.png)

#### `9920079`

![Roman truth-centered event 9920079](../assets/roman-maptruth-batch200/9920079.png)

### Events 9920080–9920089

#### `9920080`

![Roman truth-centered event 9920080](../assets/roman-maptruth-batch200/9920080.png)

#### `9920081`

![Roman truth-centered event 9920081](../assets/roman-maptruth-batch200/9920081.png)

#### `9920082`

![Roman truth-centered event 9920082](../assets/roman-maptruth-batch200/9920082.png)

#### `9920083`

![Roman truth-centered event 9920083](../assets/roman-maptruth-batch200/9920083.png)

#### `9920084`

![Roman truth-centered event 9920084](../assets/roman-maptruth-batch200/9920084.png)

#### `9920085`

![Roman truth-centered event 9920085](../assets/roman-maptruth-batch200/9920085.png)

#### `9920086`

![Roman truth-centered event 9920086](../assets/roman-maptruth-batch200/9920086.png)

#### `9920087`

![Roman truth-centered event 9920087](../assets/roman-maptruth-batch200/9920087.png)

#### `9920088`

![Roman truth-centered event 9920088](../assets/roman-maptruth-batch200/9920088.png)

#### `9920089`

![Roman truth-centered event 9920089](../assets/roman-maptruth-batch200/9920089.png)

### Events 9920090–9920098

#### `9920090`

![Roman truth-centered event 9920090](../assets/roman-maptruth-batch200/9920090.png)

#### `9920091`

![Roman truth-centered event 9920091](../assets/roman-maptruth-batch200/9920091.png)

#### `9920092`

![Roman truth-centered event 9920092](../assets/roman-maptruth-batch200/9920092.png)

#### `9920093`

![Roman truth-centered event 9920093](../assets/roman-maptruth-batch200/9920093.png)

#### `9920094`

![Roman truth-centered event 9920094](../assets/roman-maptruth-batch200/9920094.png)

#### `9920095`

![Roman truth-centered event 9920095](../assets/roman-maptruth-batch200/9920095.png)

#### `9920096`

![Roman truth-centered event 9920096](../assets/roman-maptruth-batch200/9920096.png)

#### `9920097`

![Roman truth-centered event 9920097](../assets/roman-maptruth-batch200/9920097.png)

#### `9920098`

![Roman truth-centered event 9920098](../assets/roman-maptruth-batch200/9920098.png)


### Separate validation: `9920099`

This event was inspected separately before the 199-event batch. Its figure
contains the corrected map-frame FFT/LM comparison and the truth light curve.

![Roman truth-centered validation event 9920099](../assets/roman-maptruth-batch200/9920099.png)

## Reproduction

The FFT scan uses the existing packed atlas and the truth-centered geometry
mode. The SGE wrapper is invoked once for each 100-event input/truth pair:

```bash
scripts/run-roman-maptruth-event-sge.sh \
  INPUT_ROOT TRUTH_CSV ATLAS PACKED_CACHE OUTPUT_ROOT TASK_ID
```

The direct-VBM refinement and q/s plot are reproduced with:

```bash
python scripts/roman-lm-refine.py \
  --input-root results/roman_local/roman_maptruth_fft_batch200_m128_16w1t/events \
  --output results/roman_local/roman_maptruth_lm_batch200 \
  --workers 24 --max-nfev 25 --diff-step 1e-3 --resume
python scripts/plot-roman-lm-q-s.py \
  results/roman_local/roman_maptruth_lm_batch200/summary.json \
  --pspl-truth results/roman_local/pspl_truth_dchi2_200_all_points.json \
  --dchi2-min 100 \
  --output assets/roman-lm-q-s-maptruth-dchi2gt100.png
```
