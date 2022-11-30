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


set search_path to public, eicu_crd;

drop table if exists aidxmods.heightweights;

create table aidxmods.heightweights as (

with pivot1 as (
    select 
        charttimes.uniquepid as uniquepid,
        charttimes.patienthealthsystemstayid as patienthealthsystemstayid,
        charttimes.patientunitstayid as patientunitstayid,
        charttimes.los as los,
        case
            when nursecharting.nursingchartcelltypevallabel in ('Admission Weight', 'Admit weight', 'WEIGHT in Kg')
                and nursecharting.nursingchartvalue ~ '^([0-9]+\.?[0-9]*|\.[0-9]+)$'
                and nursingchartvalue::numeric >= 1
                and nursingchartvalue::numeric <  300
                then nursingchartvalue::numeric
            else null end
        as weight,
        null::numeric as height
    
    from firstday.charttimes charttimes
    
    left join nursecharting
        on nursecharting.patientunitstayid = charttimes.patientunitstayid
        and nursecharting.nursingchartoffset >= charttimes.starttime 
        and nursecharting.nursingchartoffset <  charttimes.endtime
        and nursecharting.nursingchartcelltypecat = 'Other Vital Signs and Infusions'
),

pivot2 as (
    select 
        charttimes.uniquepid as uniquepid,
        charttimes.patienthealthsystemstayid as patienthealthsystemstayid,
        charttimes.patientunitstayid as patientunitstayid,
        charttimes.los as los,
        case
            when intakeoutput.cellpath = 'flowsheet|Flowsheet Cell Labels|I&O|Weight|Bodyweight (kg)'
                and intakeoutput.cellvaluenumeric::numeric >= 1
                and intakeoutput.cellvaluenumeric::numeric <  300
                then intakeoutput.cellvaluenumeric::numeric
            when intakeoutput.cellpath = 'flowsheet|Flowsheet Cell Labels|I&O|Weight|Bodyweight (lb)'
                and (intakeoutput.cellvaluenumeric * 0.453592)::numeric >= 1
                and (intakeoutput.cellvaluenumeric * 0.453592)::numeric <  300
                then (intakeoutput.cellvaluenumeric * 0.453592)::numeric
            else null end
        as weight,
        null::numeric as height
    
    from firstday.charttimes charttimes
    
    left join intakeoutput
        on intakeoutput.patientunitstayid = charttimes.patientunitstayid
        and intakeoutput.intakeoutputoffset >= charttimes.starttime 
        and intakeoutput.intakeoutputoffset <  charttimes.endtime
        and intakeoutput.cellpath like 'flowsheet|Flowsheet Cell Labels|I&O|Weight|Bodyweight %'
),

pivot3 as (
    select
        charttimes.uniquepid as uniquepid,
        charttimes.patienthealthsystemstayid as patienthealthsystemstayid,
        charttimes.patientunitstayid as patientunitstayid,
        charttimes.los as los,
        case
            when patient.admissionweight::numeric is not null
                and patient.admissionweight::numeric >= 1
                and patient.admissionweight::numeric <  300
                then patient.admissionweight::numeric
            else null end
        as weight,
        case
            when patient.admissionheight is not null
                and patient.admissionheight / 100 >  0.10
                and patient.admissionheight / 100 <  3.00
                then (patient.admissionheight / 100)::numeric
            else null end
        as height
    
    from firstday.charttimes charttimes
    
    left join patient
        on patient.patientunitstayid = charttimes.patientunitstayid
)
    
select
    uniquepid,
    patienthealthsystemstayid, 
    patientunitstayid,
    los,
    round(avg(weight)::numeric, 2) as weight,
    round(avg(height)::numeric, 2) as height,
    case
        when avg(height) != 0
            then round((avg(weight) / (avg(height) * avg(height)))::numeric, 2)
        else null end
    as bmi
    
from (select * from pivot1 union select * from pivot2 union select * from pivot3) pivot

group by pivot.uniquepid, pivot.patienthealthsystemstayid, pivot.patientunitstayid, pivot.los

);
