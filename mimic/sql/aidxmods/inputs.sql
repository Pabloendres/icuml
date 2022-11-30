-- --------------------------------------------------------------------------------------------------------------------
-- Inputs
-- --------------------------------------------------------------------------------------------------------------------
-- Tratamientos como infusión de vasopresores y ventilación mecánica.

-- Parámetro                        UOM                     Rango         
-- -----------------------------------------------------------------------
-- Dopamina                         mcg/kg/min              ]0, -- [
-- Epinefrina                       mcg/kg/min              ]0, -- [
-- Norepinefrina                    mcg/kg/min              ]0, -- [
-- Dobutamina                       mcg/kg/min              ]0, -- [
-- Ventilación                      bool                    bool
-- -----------------------------------------------------------------------


set search_path to public, mimiciii;

drop table if exists aidxmods.inputs;

create table aidxmods.inputs as (

select 
    charttimes.subject_id as subject_id,
    charttimes.hadm_id as hadm_id,
    charttimes.icustay_id as icustay_id, 
    charttimes.los as los,
    round(coalesce(min(dopa.vaso_rate), 0)::numeric, 2) as dopamine_min,
    round(coalesce(avg(dopa.vaso_rate), 0)::numeric, 2) as dopamine_avg,
    round(coalesce(max(dopa.vaso_rate), 0)::numeric, 2) as dopamine_max,
    round(coalesce(min(epin.vaso_rate), 0)::numeric, 2) as epinephrine_min,
    round(coalesce(avg(epin.vaso_rate), 0)::numeric, 2) as epinephrine_avg,
    round(coalesce(max(epin.vaso_rate), 0)::numeric, 2) as epinephrine_max,
    round(coalesce(min(nore.vaso_rate), 0)::numeric, 2) as norepinephrine_min,
    round(coalesce(avg(nore.vaso_rate), 0)::numeric, 2) as norepinephrine_avg,
    round(coalesce(max(nore.vaso_rate), 0)::numeric, 2) as norepinephrine_max,
    round(coalesce(min(dobu.vaso_rate), 0)::numeric, 2) as dobutamine_min,
    round(coalesce(avg(dobu.vaso_rate), 0)::numeric, 2) as dobutamine_avg,
    round(coalesce(max(dobu.vaso_rate), 0)::numeric, 2) as dobutamine_max,
    max(case
        when vent.duration_hours > 0
            then 1
        else 0 end
    ) as ventilation

from firstday.charttimes charttimes

left join dopamine_dose dopa
    on charttimes.icustay_id = dopa.icustay_id
    and (
        (charttimes.starttime, charttimes.endtime) OVERLAPS (dopa.starttime, dopa.endtime)
    )

left join epinephrine_dose epin
    on charttimes.icustay_id = epin.icustay_id
    and (
        (charttimes.starttime, charttimes.endtime) OVERLAPS (epin.starttime, epin.endtime)
    )

left join norepinephrine_dose nore
    on charttimes.icustay_id = nore.icustay_id
    and (
        (charttimes.starttime, charttimes.endtime) OVERLAPS (nore.starttime, nore.endtime)
    )

left join dobutamine_dose dobu
    on charttimes.icustay_id = dobu.icustay_id
    and (
        (charttimes.starttime, charttimes.endtime) OVERLAPS (dobu.starttime, dobu.endtime)
    )
    
left join ventilation_durations vent
    on charttimes.icustay_id = vent.icustay_id
    and (
        (charttimes.starttime, charttimes.endtime) OVERLAPS (vent.starttime, vent.endtime)
    )

group by charttimes.subject_id, charttimes.hadm_id, charttimes.icustay_id, charttimes.los
order by charttimes.subject_id, charttimes.hadm_id, charttimes.icustay_id, charttimes.los
    
);
