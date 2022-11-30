-- --------------------------------------------------------------------------------------------------------------------
-- Heightweights
-- --------------------------------------------------------------------------------------------------------------------
-- Estatura y pesos. Se calcula índice de masa corporal (IMC / BMI).

-- Parámetro                        UOM                     Rango         
-- -----------------------------------------------------------------------
-- Estatura                         m                      ]0.10, 3[      
-- Peso                             kg                     [1, 300[       
-- BMI (IMC)                        kg/m^2                 [0.1, 30000[
-- -----------------------------------------------------------------------

set search_path to public, mimiciii;

drop table if exists aidxmods.heightweights;

create table aidxmods.heightweights as (

with pivot1 as (
    select 
        charttimes.subject_id as subject_id,
        charttimes.hadm_id as hadm_id, 
        charttimes.icustay_id as icustay_id,
        charttimes.los as los,
        case
            when itemid in (762, 763, 3723, 3580, 224639, 226512)
                and valuenum >= 1
                and valuenum <  300
                then valuenum
            when itemid in (3581)
                and valuenum * 0.45359237 >= 1
                and valuenum * 0.45359237 <  300
                then valuenum * 0.45359237 -- Conv. lb to kg
            when itemid in (3582)
                and valuenum * 0.0283495231 >= 1
                and valuenum * 0.0283495231 <  300
                then valuenum * 0.0283495231 -- Conv. oz to kg
            else null end
         as weight,
         case 
            when itemid in (226730, 3485, 4188)
                and valuenum / 100 >= 0.10
                and valuenum / 100 <  3.00
                then valuenum / 100
            when itemid in (920, 1394, 4187, 3486)
                and valuenum * 2.54 / 100 >= 0.10
                and valuenum * 2.54 / 100 <  3.00
                then valuenum * 2.54 / 100 -- Conv. in to m
            else null end
         as height
    
    from firstday.charttimes charttimes
    
    left join chartevents
        on chartevents.icustay_id = charttimes.icustay_id
        and (chartevents.error is NULL or chartevents.error = 0)
        and chartevents.charttime >= charttimes.starttime
        and chartevents.charttime <  charttimes.endtime
        and chartevents.itemid in (
            762, 763, 3723, 3580, 3581, 3582, 224639, 226512,
            226730, 920, 1394, 4187, 3486, 3485, 4188
        )
),

pivot2 as (
    select 
        charttimes.subject_id as subject_id,
        charttimes.hadm_id as hadm_id, 
        charttimes.icustay_id as icustay_id,
        charttimes.los as los,
        case
            when weight is not null
                and 0.45359237 * weight >= 1
                and 0.45359237 * weight <  300
                then 0.45359237 * weight -- Conv. lb to kg
            else null end
         as weight,
         case 
            when height is not null
                and 2.54 * height / 100 >= 0.10
                and 2.54 * height / 100 <  3.00
                then 2.54 * height / 100 -- Conv. in to m
            else null end
         as height
    
    from firstday.charttimes charttimes
    
    left join echo_data
        on echo_data.hadm_id = echo_data.hadm_id
        and echo_data.hadm_id is not null
        and echo_data.charttime >= charttimes.starttime
        and echo_data.charttime <  charttimes.endtime
)

select
    subject_id,
    hadm_id, 
    icustay_id,
    los,
    round(avg(weight)::numeric, 2) as weight,
    round(avg(height)::numeric, 2) as height,
    case
        when avg(height) != 0
            then round((avg(weight) / (avg(height) * avg(height)))::numeric, 2)
        else null end
    as bmi
    
from (select * from pivot1 union select * from pivot2) pivot

    
group by pivot.subject_id, pivot.hadm_id, pivot.icustay_id, pivot.los
order by pivot.subject_id, pivot.hadm_id, pivot.icustay_id, pivot.los

);
