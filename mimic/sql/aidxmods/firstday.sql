-- --------------------------------------------------------------------------------------------------------------------
-- Adaptador "First day"
-- --------------------------------------------------------------------------------------------------------------------
-- Resumen de información sobre el primer día de ingreso a UCI.

set search_path to aidxmods;

drop table if exists aidxmods.firstday;

create table aidxmods.firstday as (

select
    -- charttimes.starttime as dt_record,
    -- NULL as recorder_place,
    -- NULL as recorder_name,
    concat('01-', charttimes.icustay_id) as idi,
    -- 'MIMIC-III' as center,
    -- 'UCI' as unit,
    admissions.age as age,
    admissions.gender as gender,
    admissions.ethnicity as ethnicity,
    heightweights.height as height,
    heightweights.weight as weight,
    -- NULL as fist_circ,
    -- NULL as body_build,
    heightweights.bmi as bmi,
    admissions.surgery as surgery,
    admissions.elective_surgery as elective_surgery,
    admissions.admission_cause as admission_cause,
    admissions.los_admission as los_admission,
    admissions.los_preicu as los_preicu,
    admissions.los_icu as los_icu,
    -- NULL as los_uti,
    -- NULL as los_emergency,
    -- admissions.admittime as dt_admission,
    -- admissions.intime as dt_icu,
    -- NULL as dt_uti,
    -- NULL as dt_emergency,
    -- admissions.dischtime as dt_discharge,
    -- admissions.deathtime as dt_death,
    admissions.nft as nft,
    admissions.death_hosp as death_hosp,
    -- NULL as death_cause,
    diagnoses.sirs as sirs,
    diagnoses.cancer as cancer,
    diagnoses.lymphoma as lymphoma,
    diagnoses.rads as rads,
    diagnoses.aids as aids,
    diagnoses.hepatic_failure as hepatic_failure,
    diagnoses.heart_failure as heart_failure,
    diagnoses.respiratory_failure as respiratory_failure,
    diagnoses.renal_failure as renal_failure,
    inputs.ventilation as ventilation,
    inputs.dopamine_min as dopamine_min,
    inputs.dopamine_avg as dopamine_avg,
    inputs.dopamine_max as dopamine_max,
    inputs.epinephrine_min as epinephrine_min,
    inputs.epinephrine_avg as epinephrine_avg,
    inputs.epinephrine_max as epinephrine_max,
    inputs.norepinephrine_min as norepinephrine_min,
    inputs.norepinephrine_avg as norepinephrine_avg,
    inputs.norepinephrine_max as norepinephrine_max,
    inputs.dobutamine_min as dobutamine_min,
    inputs.dobutamine_avg as dobutamine_avg,
    inputs.dobutamine_max as dobutamine_max,
    vitals.heartrate_min as heartrate_min,
    vitals.heartrate_avg as heartrate_avg,
    vitals.heartrate_max as heartrate_max,
    vitals.resprate_min as resprate_min,
    vitals.resprate_avg as resprate_avg,
    vitals.resprate_max as resprate_max,
    vitals.temperature_min as temperature_min,
    vitals.temperature_avg as temperature_avg,
    vitals.temperature_max as temperature_max,
    vitals.sbp_min as sbp_min,
    vitals.sbp_avg as sbp_avg,
    vitals.sbp_max as sbp_max,
    vitals.dbp_min as dbp_min,
    vitals.dbp_avg as dbp_avg,
    vitals.dbp_max as dbp_max,
    vitals.map_min as map_min,
    vitals.map_avg as map_avg,
    vitals.map_max as map_max,
    vitals.cvp_min as cvp_min,
    vitals.cvp_avg as cvp_avg,
    vitals.cvp_max as cvp_max,
    vitals.spo2_min as spo2_min,
    vitals.spo2_avg as spo2_avg,
    vitals.spo2_max as spo2_max,
    vitals.fio2_min as fio2_min,
    vitals.fio2_avg as fio2_avg,
    vitals.fio2_max as fio2_max,
    labs.pao2_min as pao2_min,
    labs.pao2_avg as pao2_avg,
    labs.pao2_max as pao2_max,
    labs.paco2_min as paco2_min,
    labs.paco2_avg as paco2_avg,
    labs.paco2_max as paco2_max,
    labs.bilirubin_min as bilirubin_min,
    labs.bilirubin_avg as bilirubin_avg,
    labs.bilirubin_max as bilirubin_max,
    labs.creatinine_min as creatinine_min,
    labs.creatinine_avg as creatinine_avg,
    labs.creatinine_max as creatinine_max,
    labs.bun_min as bun_min,
    labs.bun_avg as bun_avg,
    labs.bun_max as bun_max,
    labs.hematocrit_min as hematocrit_min,
    labs.hematocrit_avg as hematocrit_avg,
    labs.hematocrit_max as hematocrit_max,
    labs.bicarbonate_min as bicarbonate_min,
    labs.bicarbonate_avg as bicarbonate_avg,
    labs.bicarbonate_max as bicarbonate_max,
    labs.ph_min as ph_min,
    labs.ph_avg as ph_avg,
    labs.ph_max as ph_max,
    labs.platelets_min as platelets_min,
    labs.platelets_avg as platelets_avg,
    labs.platelets_max as platelets_max,
    labs.potassium_min as potassium_min,
    labs.potassium_avg as potassium_avg,
    labs.potassium_max as potassium_max,
    labs.sodium_min as sodium_min,
    labs.sodium_avg as sodium_avg,
    labs.sodium_max as sodium_max,
    labs.chloride_min as chloride_min,
    labs.chloride_avg as chloride_avg,
    labs.chloride_max as chloride_max,
    labs.magnesium_min as magnesium_min,
    labs.magnesium_avg as magnesium_avg,
    labs.magnesium_max as magnesium_max,
    labs.wbc_min as wbc_min,
    labs.wbc_avg as wbc_avg,
    labs.wbc_max as wbc_max,
    labs.lymphocytes_min as lymphocytes_min,
    labs.lymphocytes_avg as lymphocytes_avg,
    labs.lymphocytes_max as lymphocytes_max,
    labs.neutrophils_min as neutrophils_min,
    labs.neutrophils_avg as neutrophils_avg,
    labs.neutrophils_max as neutrophils_max,
    labs.ast_min as ast_min,
    labs.ast_avg as ast_avg,
    labs.ast_max as ast_max,
    labs.alt_min as alt_min,
    labs.alt_avg as alt_avg,
    labs.alt_max as alt_max,
    labs.alp_min as alp_min,
    labs.alp_avg as alp_avg,
    labs.alp_max as alp_max,
    labs.albumin_min as albumin_min,
    labs.albumin_avg as albumin_avg,
    labs.albumin_max as albumin_max,
    labs.glucose_min as glucose_min,
    labs.glucose_avg as glucose_avg,
    labs.glucose_max as glucose_max,
    labs.base_excess_min as base_excess_min,
    labs.base_excess_avg as base_excess_avg,
    labs.base_excess_max as base_excess_max,
    labs.ptt_min as ptt_min,
    labs.ptt_avg as ptt_avg,
    labs.ptt_max as ptt_max,
    vitals.bnp_min as bnp_min,
    vitals.bnp_avg as bnp_avg,
    vitals.bnp_max as bnp_max,
    coalesce(labs.fibrinogen_min, vitals.fibrinogen_min) as fibrinogen_min,
    coalesce(labs.fibrinogen_avg, vitals.fibrinogen_avg) as fibrinogen_avg,
    coalesce(labs.fibrinogen_max, vitals.fibrinogen_max) as fibrinogen_max,
    labs.hemoglobin_min as hemoglobin_min,
    labs.hemoglobin_avg as hemoglobin_avg,
    labs.hemoglobin_max as hemoglobin_max,
    labs.lactate_min as lactate_min,
    labs.lactate_avg as lactate_avg,
    labs.lactate_max as lactate_max,
    gcs.gcs_min as gcs_min,
    gcs.gcs_avg as gcs_avg,
    gcs.gcs_max as gcs_max,
    outputs.urineoutput as urineoutput
    
from charttimes

left join admissions
    on admissions.hadm_id = charttimes.hadm_id

left join diagnoses
    on diagnoses.hadm_id = charttimes.hadm_id
    
left join heightweights
    on heightweights.icustay_id = charttimes.icustay_id
    and heightweights.los = charttimes.los

left join vitals
    on vitals.icustay_id = charttimes.icustay_id
    and vitals.los = charttimes.los
    
left join labs
    on labs.icustay_id = charttimes.icustay_id
    and labs.los = charttimes.los
    
left join gcs
    on gcs.icustay_id = charttimes.icustay_id
    and gcs.los = charttimes.los
    
left join inputs
    on inputs.icustay_id = charttimes.icustay_id
    and inputs.los = charttimes.los
    
left join outputs
    on outputs.icustay_id = charttimes.icustay_id
    and outputs.los = charttimes.los

order by charttimes.subject_id, charttimes.hadm_id, charttimes.icustay_id, charttimes.los
    
);
