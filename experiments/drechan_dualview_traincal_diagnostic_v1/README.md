# Frozen DREChannNet TRAIN/CAL diagnosis

This is a zero-training diagnostic after the fixed dual-view candidate's failed performance gate. Source is pushed before execution. There are no actual diagnostic outcomes yet. The original80-patientOuterMacro-F1=0.653828remains unchanged.

The completed dual-view study improved ranking but lost Macro-F1. Historical Raw-only margin-grid,patient-scoreZ andfixed-fraction decoders did not yield reliable gains;historicalA0posterior identification also failed. No decoder rescue or new density model is assumed. The present study asks whether selected bodies show initial-anchor latent drift andCALthreshold sensitivity,using legal TRAIN/CAL only.

The[locked protocol](PROTOCOL_LOCK.json)defines45selected calibration cells,30complete TRAIN/CALlatent forward bodies,60latent phase cells and75unlabeled score phase cells. Five changed critical-path tests precede execution. All45derived private caches seal before aggregate reporting. An independent verifier uses the original selector and sklearn confusion metrics to check360leave-oneCAL cases and all descriptive summaries.

Leave-oneCAL excludes a patient's labels only from threshold fitting,not from the already completed checkpoint selection. It is therefore selection-conditioned sensitivity,not unbiased cross-validation or new validation performance. No new threshold is deployed. No assessment/Development/Outer predictions or metrics are read or generated. No neural weight,cache source,feature,patient role or existing result is changed.

Run`run_all.ps1 -RunTag <fresh> -SourceCommit <verified40hex>`under a sustained foreground SSH owner on the trusted server. Code/runtime stay on Ddrive andexisting raw caches are read in place. Native or Python errors stop for diagnosis,not blind retry. Only anonymous aggregate tables,source andhashes may be exported. Derived vectors/logits,identities,weights,raw/cache andlogs stay on the server. This diagnosis alone cannot qualify a new main model or achieve the active goal.
