-- --------------------------------------------------------------------------------------------------------------------
-- Outputs
-- --------------------------------------------------------------------------------------------------------------------
-- Signos vitales.

-- Parámetro                        UOM                     Rango         
-- -----------------------------------------------------------------------
-- Salida de orina                  mL                      [0, -- [
-- -----------------------------------------------------------------------


set search_path to public, mimiciii;

drop table if exists aidxmods.outputs;

create table aidxmods.outputs as (

with pivot as (
    select 
        charttimes.subject_id as subject_id,
        charttimes.hadm_id as hadm_id, 
        charttimes.icustay_id as icustay_id,
        charttimes.los as los,
        (case
            when outputevents.itemid = 227488
                and outputevents.value > 0
                 then (-1 * outputevents.value)::numeric -- Negative value for this one
            else outputevents.value::numeric end
         ) as urineoutput

    FROM firstday.charttimes charttimes

    left join outputevents
        on outputevents.icustay_id = charttimes.icustay_id
        and outputevents.charttime >= charttimes.starttime
        and outputevents.charttime <  charttimes.endtime
        and outputevents.itemid in (
            40055, -- "Urine Out Foley"
            43175, -- "Urine ."
            40069, -- "Urine Out Void"
            40094, -- "Urine Out Condom Cath"
            40715, -- "Urine Out Suprapubic"
            40473, -- "Urine Out IleoConduit"
            40085, -- "Urine Out Incontinent"
            40057, -- "Urine Out Rt Nephrostomy"
            40056, -- "Urine Out Lt Nephrostomy"
            40405, -- "Urine Out Other"
            40428, -- "Urine Out Straight Cath"
            40086, -- Urine Out Incontinent
            40096, -- "Urine Out Ureteral Stent #1"
            40651, -- "Urine Out Ureteral Stent #2"
            226559, -- "Foley"
            226560, -- "Void"
            226561, -- "Condom Cath"
            226584, -- "Ileoconduit"
            226563, -- "Suprapubic"
            226564, -- "R Nephrostomy"
            226565, -- "L Nephrostomy"
            226567, -- Straight Cath
            226557, -- R Ureteral Stent
            226558, -- L Ureteral Stent
            227488, -- GU Irrigant Volume In
            227489  -- GU Irrigant/Urine Volume Out    
        )
)

select 
    subject_id, 
    hadm_id,
    icustay_id,
    los,
    round(sum(urineoutput)::numeric, 2) as urineoutput
    
from pivot
    
group by pivot.subject_id, pivot.hadm_id, pivot.icustay_id, pivot.los
order by pivot.subject_id, pivot.hadm_id, pivot.icustay_id, pivot.los
    
);
