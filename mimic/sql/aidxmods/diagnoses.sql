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

set search_path to public, mimiciii;

drop table if exists aidxmods.diagnoses;

create table aidxmods.diagnoses as (

with pivot as (
    select
        subject_id,
        hadm_id,
        max(case
            when   icd9_code like '456%'
                or icd9_code like '56723%'
                or icd9_code like '5712%'
                or icd9_code like '5715%'
                or icd9_code like '5722%'
                or icd9_code like '5724%'
                or icd9_code like '78959%'
                or icd9_code like '53082%'
                or icd9_code like '5789%'
                then 1
            else 0 end
            ) as cirrhosis,
        max(case
            when   icd9_code like '5723%'
                then 1
            else 0 end
        ) as portal_hypertension,
        max(case
            when   icd9_code like '0702%'
                or icd9_code like '0703%'
                or icd9_code like '07041%'
                or icd9_code like '07044%'
                or icd9_code like '07051%'
                or icd9_code like '07054%'
                or icd9_code like '0707%'
                or icd9_code like '155%'
                or icd9_code like '275%'
                or icd9_code like '571%'
                or icd9_code like '5733%'
                or icd9_code like '5722%'
                then 1
            else 0 end
        ) as hepatic_failure,
        max(case
            when   icd9_code like '428%'
                then 1
            else 0 end
        ) as heart_failure,
        max(case
            when   icd9_code like '490%'
                or icd9_code like '491%'
                or icd9_code like '492%'
                or icd9_code like '4930%'
                or icd9_code like '4931%'
                or icd9_code like '4932%'
                or icd9_code like '4938%'
                or icd9_code like '4939%'
                or icd9_code like '494%'
                or icd9_code like '495%'
                or icd9_code like '496%'
                or icd9_code like '500%'
                or icd9_code like '501%'
                or icd9_code like '502%'
                or icd9_code like '503%'
                or icd9_code like '504%'
                or icd9_code like '505%'
                or icd9_code like '5064%'
                or icd9_code like '440%'
                or icd9_code like '4412%'
                or icd9_code like '4414%'
                or icd9_code like '4417%'
                or icd9_code like '4419%'
                or icd9_code like '4431%'
                or icd9_code like '4432%'
                or icd9_code like '4438%'
                or icd9_code like '4439%'
                or icd9_code like '4471%'
                or icd9_code like '5571%'
                or icd9_code like '5579%'
                or icd9_code like 'V434%'
                or icd9_code like '51883%'
                or icd9_code like '79902%'
                or icd9_code like '78609%'
                or icd9_code like '2890%'
                or icd9_code like '416%'
                or icd9_code like 'V4611%'
                then 1
            else 0 end
        ) as respiratory_failure,
        max(case
            when   icd9_code like '5856%'
                then 1
                else 0 end
        ) as renal_failure,
        max(case
            when   icd9_code like 'V581%'
                or icd9_code like 'V5865%'
                or icd9_code like 'V8741%'
                or icd9_code like 'E9331%'
                or icd9_code like 'E926%'
                then 1
            else 0 end
        ) as rads,
        max(case
            when   icd9_code like '196%'
                or icd9_code like '197%'
                or icd9_code like '198%'
                or icd9_code like '1990%'
                or icd9_code like '1991%'
                then 1
            else 0 end
        ) as cancer,
        max(case
            when   icd9_code like '200%'
                or icd9_code like '201%'
                or icd9_code like '2020%'
                or icd9_code like '2021%'
                or icd9_code like '2022%'
                or icd9_code like '2023%'
                or icd9_code like '2025%'
                or icd9_code like '2026%'
                or icd9_code like '2027%'
                or icd9_code like '2028%'
                or icd9_code like '2029%'
                or icd9_code like '203%'
                or icd9_code like '20300%'
                or icd9_code like '20301%'
                or icd9_code like '20380%'
                or icd9_code like '20381%'
                or icd9_code like '204%'
                or icd9_code like '205%'
                or icd9_code like '206%'
                or icd9_code like '207%'
                or icd9_code like '208%'
                or icd9_code like '2386%'
                or icd9_code like '2733%'
                or icd9_code like 'V1071%'
                or icd9_code like 'V1072%'
                or icd9_code like 'V1079%'
                then 1
            else 0 end
        ) as lymphoma,
        max(case
            when   icd9_code like '042%'
                or icd9_code like '176%'
                or icd9_code like '1363%'
                or icd9_code like '130%'
                or icd9_code like '010%'
                or icd9_code like '011%'
                or icd9_code like '012%'
                or icd9_code like '013%'
                or icd9_code like '014%'
                or icd9_code like '015%'
                or icd9_code like '016%'
                or icd9_code like '017%'
                or icd9_code like '018%'
                then 1
            else 0 end
        ) as aids,
        max(case
            when   icd9_code like '9959%'
                then 1
            else 0 end
        ) as sirs

    from diagnoses_icd

    group by subject_id, hadm_id
)

select
    charttimes.subject_id,
    charttimes.hadm_id,
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

left join pivot on pivot.hadm_id = charttimes.hadm_id
    
);
