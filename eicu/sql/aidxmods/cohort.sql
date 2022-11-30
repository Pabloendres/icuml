-- --------------------------------------------------------------------------------------------------------------------
-- Cohort
-- --------------------------------------------------------------------------------------------------------------------
-- Estudio de cohorte (observaciones excluidas).

-- Criterios:
-- 1. Primer ingreso a UCI de cada paciente (first_time).
-- 2. Pacientes de al menos 1 año de edad (one_yo).
-- 3. Estancia en UCI mayor a 24 horas (los_24h).
-- 
--  total  | first_time | excl_first_time | one_yo | excl_one_yo | los_24h | excl_los_24h | full_therapy | excl_full_therapy 
-- --------+------------+-----------------+--------+-------------+---------+--------------+--------------+-------------------
--  200859 |     132574 |           68285 | 132557 |          17 |   95555 |        37002 |        82413 |             13142
-- 

set search_path to public, eicu_crd;

-- Universo inicial
with excluded0 as (
    select
        uniquepid,
        patienthealthsystemstayid,
        patientunitstayid,
        row_number() over (
            partition by uniquepid
            order by unitdischargeoffset desc
        ) as max_los,
        1 as los,
        0 as starttime,
        1440 as endtime

    from patient
    
    where true
),

-- Excluir ingresos secundarios por paciente.
excluded1 as (
    select
        uniquepid,
        patienthealthsystemstayid,
        patientunitstayid,
        row_number() over (
            partition by uniquepid
            order by unitdischargeoffset desc
        ) as max_los,
        1 as los,
        0 as starttime,
        1440 as endtime

    from patient

    where
        hospitaladmitoffset <= 0
        and unitvisitnumber = 1
        --and age <> '0'
        --and unitdischargeoffset >= 1440
),

-- Excluir pacientes menores a un año
excluded2 as (
    select
        uniquepid,
        patienthealthsystemstayid,
        patientunitstayid,
        row_number() over (
            partition by uniquepid
            order by unitdischargeoffset desc
        ) as max_los,
        1 as los,
        0 as starttime,
        1440 as endtime

    from patient

    where
        hospitaladmitoffset <= 0
        and unitvisitnumber = 1
        and age <> '0'
        --and unitdischargeoffset >= 1440
),


-- Excluir estancias inferiores a 24 horas
excluded3 as (
    select
        uniquepid,
        patienthealthsystemstayid,
        patientunitstayid,
        row_number() over (
            partition by uniquepid
            order by unitdischargeoffset desc
        ) as max_los,
        1 as los,
        0 as starttime,
        1440 as endtime

    from patient

    where
        hospitaladmitoffset <= 0
        and unitvisitnumber = 1
        and age <> '0'
        and unitdischargeoffset >= 1440
),

-- Excluir pacientes sin terapia completa
excluded4 as (
    select
        idi, nft
    from aidxmods.firstday
)
   
select

    (select count(*) from excluded0) as total,
    (select count(*) from excluded1 where excluded1.max_los = 1) as first_time,
    (select count(*) from excluded0) - (select count(*) from excluded1 where excluded1.max_los = 1) as excl_first_time,
    (select count(*) from excluded2 where excluded2.max_los = 1) as one_yo,
    (select count(*) from excluded1 where excluded1.max_los = 1) - (select count(*) from excluded2 where excluded2.max_los = 1) as excl_one_yo,
    (select count(*) from excluded3 where excluded3.max_los = 1) as los_24h,
    (select count(*) from excluded2 where excluded2.max_los = 1) - (select count(*) from excluded3 where excluded3.max_los = 1) as excl_los_24h,
    (select count(*) from excluded4 where nft = 0) as full_therapy,
    (select count(*) from excluded4 where nft = 1) as excl_full_therapy;
