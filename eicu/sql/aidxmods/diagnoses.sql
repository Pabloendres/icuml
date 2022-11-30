-- --------------------------------------------------------------------------------------------------------------------
-- Diagnoses
-- --------------------------------------------------------------------------------------------------------------------
-- Diagnósticos ICD-9. Diagnósticos incluídos (consultar documentación por definiciones):
-- 1. Falla hepática.
-- 2. Falla cardiaca.
-- 3. Falla respiratoria.
-- 4. Falla renal.
-- 5. Inmunosupresión por radiación.
-- 6. Cáncer.
-- 7. Linfoma.
-- 8. SIDA.
-- 9. SIRS.


set search_path to public, eicu_crd;

drop table if exists aidxmods.diagnoses;

create table aidxmods.diagnoses as (

with pivot as (
    select
        patientunitstayid,
        max(case
            when   icd9code like '%456.%'
                or icd9code like '%567.23%'
                or icd9code like '%571.2%'
                or icd9code like '%571.5%'
                or icd9code like '%572.2%'
                or icd9code like '%572.4%'
                or icd9code like '%789.59%'
                or icd9code like '%530.82%'
                or icd9code like '%578.9%'
                then 1
            else 0 end
            ) as cirrhosis,
        max(case
            when   icd9code like '%572.3%'
                then 1
            else 0 end
        ) as portal_hypertension,
        max(case
            when   icd9code like '%070.2%'
                or icd9code like '%070.3%'
                or icd9code like '%070.41%'
                or icd9code like '%070.44%'
                or icd9code like '%070.51%'
                or icd9code like '%070.54%'
                or icd9code like '%070.7%'
                or icd9code like '%155%'
                or icd9code like '%275%'
                or icd9code like '%571%'
                or icd9code like '%573.3%'
                or icd9code like '%572.2%'
                then 1
            else 0 end
        ) as hepatic_failure,
        max(case
            when   icd9code like '%428%'
                then 1
            else 0 end
        ) as heart_failure,
        max(case
            when   icd9code like '%490%'
                or icd9code like '%491%'
                or icd9code like '%492%'
                or icd9code like '%493.0%'
                or icd9code like '%493.1%'
                or icd9code like '%493.2%'
                or icd9code like '%493.8%'
                or icd9code like '%493.9%'
                or icd9code like '%494%'
                or icd9code like '%495%'
                or icd9code like '%496%'
                or icd9code like '%500%'
                or icd9code like '%501%'
                or icd9code like '%502%'
                or icd9code like '%503%'
                or icd9code like '%504%'
                or icd9code like '%505%'
                or icd9code like '%506.4%'
                or icd9code like '%440%'
                or icd9code like '%441.2%'
                or icd9code like '%441.4%'
                or icd9code like '%441.7%'
                or icd9code like '%441.9%'
                or icd9code like '%443.1%'
                or icd9code like '%443.2%'
                or icd9code like '%443.8%'
                or icd9code like '%443.9%'
                or icd9code like '%447.1%'
                or icd9code like '%557.1%'
                or icd9code like '%557.9%'
                or icd9code like '%V43.4%'
                or icd9code like '%518.83%'
                or icd9code like '%799.02%'
                or icd9code like '%786.09%'
                or icd9code like '%289.0%'
                or icd9code like '%416%'
                or icd9code like '%V46.11%'
                then 1
            else 0 end
        ) as respiratory_failure,
        max(case
            when   icd9code like '%585.6%'
                then 1
            else 0 end
        ) as renal_failure,
        max(case
            when   icd9code like '%V58.1%'
                or icd9code like '%V58.65%'
                or icd9code like '%V87.41%'
                or icd9code like '%E933.1%'
                or icd9code like '%E926%'
                then 1
            else 0 end
        ) as rads,
        max(case
            when   icd9code like '%196%'
                or icd9code like '%197%'
                or icd9code like '%198%'
                or icd9code like '%199.0%'
                or icd9code like '%199.1%'
                then 1
            else 0 end
        ) as cancer,
        max(case
            when   icd9code like '%200%'
                or icd9code like '%201%'
                or icd9code like '%202.0%'
                or icd9code like '%202.1%'
                or icd9code like '%202.2%'
                or icd9code like '%202.3%'
                or icd9code like '%202.5%'
                or icd9code like '%202.6%'
                or icd9code like '%202.7%'
                or icd9code like '%202.8%'
                or icd9code like '%202.9%'
                or icd9code like '%203%'
                or icd9code like '%203.00%'
                or icd9code like '%203.01%'
                or icd9code like '%203.80%'
                or icd9code like '%203.81%'
                or icd9code like '%204%'
                or icd9code like '%205%'
                or icd9code like '%206%'
                or icd9code like '%207%'
                or icd9code like '%208%'
                or icd9code like '%238.6%'
                or icd9code like '%273.3%'
                or icd9code like '%V10.71%'
                or icd9code like '%V10.72%'
                or icd9code like '%V10.79%'
                then 1
            else 0 end
        ) as lymphoma,
        max(case
            when   icd9code like '%042%'
                or icd9code like '%176%'
                or icd9code like '%136.3%'
                or icd9code like '%130%'
                or icd9code like '%010%'
                or icd9code like '%011%'
                or icd9code like '%012%'
                or icd9code like '%013%'
                or icd9code like '%014%'
                or icd9code like '%015%'
                or icd9code like '%016%'
                or icd9code like '%017%'
                or icd9code like '%018%'
                then 1
                else 0 end
        ) as aids,
        max(case
            when   icd9code like '%995.9%'
                then 1
            else 0 end
        ) as sirs


    from diagnosis

    group by patientunitstayid
)

select
    charttimes.uniquepid,
    charttimes.patienthealthsystemstayid,
    case
        when pivot.hepatic_failure = 1
            or (pivot.cirrhosis = 1
            and pivot.portal_hypertension = 1)
        then 1
        else 0 end
    as hepatic_failure,
    pivot.heart_failure,
    pivot.respiratory_failure,
    pivot.renal_failure,
    pivot.rads,
    pivot.cancer,
    pivot.lymphoma,
    pivot.aids,
    pivot.sirs

from firstday.charttimes charttimes

left join pivot on pivot.patientunitstayid = charttimes.patientunitstayid

);
