-- --------------------------------------------------------------------------------------------------------------------
-- Admissions
-- --------------------------------------------------------------------------------------------------------------------
-- Información de admisión y alta de hospital y UCI.

-- Parámetro                        UOM                     Rango         
-- -----------------------------------------------------------------------
-- Causa de admisión                --                      --
-- Cirugía                          bool                    bool
-- Cirugia electiva                 bool                    bool
-- Fecha de admisión                fecha                   --
-- Fecha de ingreso UCI             fecha                   --
-- Fecha de egreso UCI              fecha                   --
-- Fecha de alta                    fecha                   --
-- Estadía pre UCI                  días                    [0, --[
-- Estadía UCI                      días                    [0, --[
-- Estadía post UCI                 días                    [0, --[
-- Estadía admisión                 días                    [0, --[
-- Estadía admisión a egreso UCI    días                    [0, --[
-- N° de ingreso a UCI              --                      [1, --[
-- Edad                             años                    ]1, 90]
-- Sexo                             --                      --
-- Etnia                            --                      --
-- Fecha de muerte                  fecha                   bool
-- Muerte hospitalaria              bool                    bool
-- -----------------------------------------------------------------------

set search_path to public, mimiciii;

drop table if exists aidxmods.admissions;

create table aidxmods.admissions as (
    
with pivot1 as (
    select
        admissions.subject_id as subject_id, 
        admissions.hadm_id as hadm_id,
        admissions.diagnosis as admission_cause,
        admissions.admittime as admittime,
        icustay_detail.intime as intime,
        icustay_detail.outtime as outtime,
        admissions.dischtime as dischtime,
        extract(epoch from icustay_detail.intime - admissions.admittime) / 86400 as los_preicu, 
        extract(epoch from icustay_detail.outtime - icustay_detail.intime) / 86400 as los_icu,
        extract(epoch from admissions.dischtime - icustay_detail.outtime) / 86400 as los_posticu,
        extract(epoch from admissions.dischtime - admissions.admittime) / 86400 as los_admission,
        extract(epoch from icustay_detail.outtime - admissions.admittime) / 86400 as los_icudischarge,
        icustay_detail.icustay_seq as icustay_seq,
        case
            when icustay_detail.admission_age > 90
                then 90
            else icustay_detail.admission_age::int end
        as age,
        case
            when icustay_detail.gender in ('FEMALE', 'Female', 'female', 'F')
                then 0
            when icustay_detail.gender in ('MALE', 'Male', 'male', 'M')
                then 1
            else null end
        as gender,
        case
            when icustay_detail.ethnicity in 
                ('Caucasian','WHITE','WHITE - BRAZILIAN','WHITE - EASTERN EUROPEAN','WHITE - OTHER EUROPEAN','WHITE - RUSSIAN')
                then 'White'
            when icustay_detail.ethnicity in 
                ('Hispanic','CARIBBEAN ISLAND','HISPANIC OR LATINO','HISPANIC/LATINO - CENTRAL AMERICAN (OTHER)', 'HISPANIC/LATINO - COLOMBIAN',
                 'HISPANIC/LATINO - CUBAN','HISPANIC/LATINO - DOMINICAN','HISPANIC/LATINO - GUATEMALAN','HISPANIC/LATINO - HONDURAN',
                'HISPANIC/LATINO - MEXICAN','HISPANIC/LATINO - PUERTO RICAN','HISPANIC/LATINO - SALVADORAN','SOUTH AMERICAN')
                then 'Hispanic'
            when icustay_detail.ethnicity in 
                ('African American','BLACK/AFRICAN','BLACK/AFRICAN AMERICAN','BLACK/CAPE VERDEAN','BLACK/HAITIAN')
                then 'African American'
            when icustay_detail.ethnicity in 
                ('Asian','ASIAN','ASIAN - ASIAN INDIAN','ASIAN - CAMBODIAN','ASIAN - CHINESE','ASIAN - FILIPINO','ASIAN - JAPANESE',
                 'ASIAN - KOREAN','ASIAN - OTHER', 'ASIAN - THAI','ASIAN - VIETNAMESE')
                then 'Asian'
            when icustay_detail.ethnicity in 
                ('Native American', 'Other/Unknown','AMERICAN INDIAN/ALASKA NATIVE','AMERICAN INDIAN/ALASKA NATIVE FEDERALLY RECOGNIZED TRIBE',
                 'MIDDLE EASTERN','MULTI RACE ETHNICITY','NATIVE HAWAIIAN OR OTHER PACIFIC ISLANDER','OTHER','PATIENT DECLINED TO ANSWER',
                 'UNABLE TO OBTAIN','UNKNOWN/NOT SPECIFIED','PORTUGUESE')
                then 'Other/Unknown'
            else 'Other/Unknown' end
        as ethnicity,
        admissions.deathtime as deathtime,
        admissions.hospital_expire_flag as death_hosp

    from admissions
    
    left join icustay_detail
        on icustay_detail.hadm_id = admissions.hadm_id
    
    where 
        icustay_detail.first_hosp_stay = 't' 
        and icustay_detail.first_icu_stay = 't'
        and icustay_detail.admission_age > 1
        and los_icu >= 1
),
    
pivot2 as (
    select
        admissions.subject_id as subject_id, 
        admissions.hadm_id as hadm_id,
        max(case
            when lower(services.curr_service) like '%surg%'
                then 1
            else 0 end
        )as surgery,
        max(case
            when admissions.admission_type = 'ELECTIVE'
                then 1
            else 0 end
        ) as elective_surgery,
        max(case
            when chartevents.value not in ('Full code', 'Full Code', 'CPR Not Indicate')
                and chartevents.charttime < icustay_detail.intime + interval '4 day'
                and chartevents.icustay_id = icustay_detail.icustay_id
                then 1
            else 0 end
        ) as nft
    
    from admissions
    
    left join icustay_detail
        on icustay_detail.hadm_id = admissions.hadm_id

    left join services
        on services.hadm_id = admissions.hadm_id
        and services.transfertime >= icustay_detail.intime
        and services.transfertime <  icustay_detail.outtime
    
    left join chartevents
        on chartevents.hadm_id = admissions.hadm_id
        and (chartevents.error is NULL or chartevents.error = 0)
        and chartevents.itemid in (
            128,    -- Code Status                           | carevue
            223758  -- Code Status                           | metavision
        )
    
    where 
        icustay_detail.first_hosp_stay = 't' 
        and icustay_detail.first_icu_stay = 't'
        and icustay_detail.admission_age > 1
        and los_icu >= 1
    
    group by admissions.subject_id, admissions.hadm_id
)

select
    pivot1.subject_id, 
    pivot1.hadm_id,
    pivot1.admission_cause,
    pivot2.surgery,
    pivot2.elective_surgery,
    pivot1.admittime,
    pivot1.intime,
    pivot1.outtime,
    pivot1.dischtime,
    round(pivot1.los_preicu::numeric, 2) as los_preicu, 
    round(pivot1.los_icu::numeric, 2) as los_icu,
    round(pivot1.los_posticu::numeric, 2) as los_posticu,
    round(pivot1.los_admission::numeric, 2) as los_admission,
    round(pivot1.los_icudischarge::numeric, 2) as los_icudischarge,
    pivot1.icustay_seq,
    pivot1.age,
    pivot1.gender,
    pivot1.ethnicity,
    pivot2.nft,
    pivot1.deathtime,
    pivot1.death_hosp
    
from pivot1
    
left join pivot2
    on pivot2.hadm_id = pivot1.hadm_id
    
);
