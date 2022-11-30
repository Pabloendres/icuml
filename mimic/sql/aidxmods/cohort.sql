-- --------------------------------------------------------------------------------------------------------------------
-- Cohort
-- --------------------------------------------------------------------------------------------------------------------
-- Estudio de cohorte (observaciones excluidas).

-- Criterios:
-- 1. Primer ingreso a UCI de cada paciente (first_time).
-- 2. Pacientes de al menos 1 año de edad (one_yo).
-- 3. Estancia en UCI mayor a 24 horas (los_24h).

--  total | first_time | excl_first_time | one_yo | excl_one_yo | los_24h | excl_los_24h | full_therapy | excl_full_therapy 
-- -------+------------+-----------------+--------+-------------+---------+--------------+--------------+-------------------
--  61051 |      46428 |           14623 |  38566 |        7862 |   32629 |         5937 |        29672 |              2957




set search_path to public, mimiciii;

with excluded0 as (
    select
        subject_id,
        hadm_id,
        icustay_id,
        1 as los,
        date_trunc('day', intime) as starttime,
        date_trunc('day', intime) + interval '1 day' as endtime

    from icustay_detail
),

excluded1 as (
    select
        subject_id,
        hadm_id,
        icustay_id,
        1 as los,
        date_trunc('day', intime) as starttime,
        date_trunc('day', intime) + interval '1 day' as endtime

    from icustay_detail

    where 
        first_hosp_stay = 't' 
        and first_icu_stay = 't'
),

excluded2 as (
    select
        subject_id,
        hadm_id,
        icustay_id,
        1 as los,
        date_trunc('day', intime) as starttime,
        date_trunc('day', intime) + interval '1 day' as endtime

    from icustay_detail

    where 
        first_hosp_stay = 't' 
        and first_icu_stay = 't'
        and admission_age > 1
),

excluded3 as (
    select
        subject_id,
        hadm_id,
        icustay_id,
        1 as los,
        date_trunc('day', intime) as starttime,
        date_trunc('day', intime) + interval '1 day' as endtime

    from icustay_detail

    where 
        first_hosp_stay = 't' 
        and first_icu_stay = 't'
        and admission_age > 1
        and los_icu >= 1
),

-- Excluir pacientes sin terapia completa
excluded4 as (
    select
        idi, nft
    from aidxmods.firstday
)

select

    (select count(*) from excluded0) as total,
    (select count(*) from excluded1) as first_time,
    (select count(*) from excluded0) - (select count(*) from excluded1) as excl_first_time,
    (select count(*) from excluded2) as one_yo,
    (select count(*) from excluded1) - (select count(*) from excluded2) as excl_one_yo,
    (select count(*) from excluded3) as los_24h,
    (select count(*) from excluded2) - (select count(*) from excluded3) as excl_los_24h,
    (select count(*) from excluded4 where nft = 0) as full_therapy,
    (select count(*) from excluded4 where nft = 1) as excl_full_therapy;
