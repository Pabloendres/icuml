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


set search_path to public, mimiciii;

drop table if exists aidxmods.charttimes;

create table aidxmods.charttimes as (

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
);
