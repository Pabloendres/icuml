-- --------------------------------------------------------------------------------------------------------------------
-- Charttimes
-- --------------------------------------------------------------------------------------------------------------------
-- Pivote principal para la creación de tablas.

-- Criterios:
-- 1. Resumir primeras 24 horas de ingreso a UCI del primer ingreso a UCI de cada paciente.
-- 2. Pacientes de al menos 1 año.
-- 3. Estancia en UCI mayor a 24 horas.

-- Parámetro                        UOM                     Rango         
-- -----------------------------------------------------------------------
-- Estadía                          Días                    {1}
-- Estadía de inicio                Horas                   [0, --[
-- Estadía de término               Horas                   [0, --[
-- -----------------------------------------------------------------------


set search_path to public, eicu_crd;

drop table if exists aidxmods.charttimes;

create table aidxmods.charttimes as (

with pivot as (
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
)
    
select
    uniquepid,
    patienthealthsystemstayid,
    patientunitstayid,
    los,
    starttime,
    endtime
    
from pivot
    
where max_los = 1
    
order by uniquepid, patienthealthsystemstayid, patientunitstayid, los, starttime
);
