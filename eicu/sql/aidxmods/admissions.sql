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


set search_path to public, eicu_crd;

drop table if exists aidxmods.admissions;

create table aidxmods.admissions as (
    
with pivot1 as (
    select
        patient.uniquepid, 
        patient.patienthealthsystemstayid,
        patient.apacheadmissiondx as admission_cause,
        case
            when patient.unitadmitsource in ('Recovery Room', 'PACU', 'Operating Room')
                then 1
            else 0 end
        as surgery,
        case
            when apachepredvar.electivesurgery is not null
                then apachepredvar.electivesurgery
            else 0 end
        as elective_surgery,
        '2100-01-01'::date + patient.hospitaladmitoffset / 1440 as admittime,
        '2100-01-01'::date + 0 as intime,
        '2100-01-01'::date + patient.unitdischargeoffset / 1440 as outtime,
        '2100-01-01'::date + patient.hospitaldischargeoffset / 1440 as dischtime,
        - patient.hospitaladmitoffset / 1440 as los_preicu, 
        patient.unitdischargeoffset / 1440 as los_icu,
        (patient.hospitaldischargeoffset - patient.unitdischargeoffset) / 1440 as los_posticu,
        (patient.hospitaldischargeoffset - patient.hospitaladmitoffset) / 1440 as los_admission,
        (patient.unitdischargeoffset - patient.hospitaladmitoffset) / 1440 as los_icudischarge,
        patient.unitvisitnumber as icustay_seq,
        case
            when patient.age = '> 89'
                then 90
            when patient.age ~ '^[0-9]+[.]?[0-9]*$'
                then patient.age::int
            else null end
        as age,
        case
            when patient.gender in ('FEMALE', 'Female', 'female', 'F')
                then 0
            when patient.gender in ('MALE', 'Male', 'male', 'M')
                then 1
            else null end
        as gender,
        case
            when patient.ethnicity = 'Caucasian'
                then 'White'
            when patient.ethnicity = 'Hispanic'
                then 'Hispanic'
            when patient.ethnicity = 'African American'
                then 'African American'
            when patient.ethnicity = 'Asian'
                then 'Asian'
            when patient.ethnicity in ('Native American', 'Other/Unknown')
                then 'Other/Unknown'
            else 'Other/Unknown' end
        as ethnicity,
        case
            when patient.unitdischargestatus = 'Expired'
                then '2100-01-01'::date + patient.unitdischargeoffset / 1440
            else null end
        as deathtime,
        case
            when patient.unitdischargestatus = 'Expired'
                then 1
            else 0 end
        as death_hosp

    from firstday.charttimes
    
    left join patient
        on patient.patientunitstayid = charttimes.patientunitstayid
    
    left join apachepredvar
        on apachepredvar.patientunitstayid = charttimes.patientunitstayid
    
    where
        patient.hospitaladmitoffset <= 0
        and patient.unitvisitnumber = 1
        and patient.age <> '0'
        and patient.unitdischargeoffset >= 1440
),
    
pivot2 as (
    select
        charttimes.uniquepid,
        charttimes.patienthealthsystemstayid,
        max(case
            when careplangeneral.cplitemvalue not in ('Full therapy', 'No vasopressors/inotropes')
                and careplangeneral.cplitemoffset < 1440 * 4
                and careplangeneral.patientunitstayid = charttimes.patientunitstayid
                then 1
            else 0 end
        ) as nft
    
    from firstday.charttimes

    left join patient
        on patient.patientunitstayid = charttimes.patientunitstayid

    left join careplangeneral
        on careplangeneral.patientunitstayid = charttimes.patientunitstayid
        and careplangeneral.cplgroup = 'Care Limitation'

    where
        patient.hospitaladmitoffset <= 0
        and patient.unitvisitnumber = 1
        and patient.age <> '0'
        and patient.unitdischargeoffset >= 1440
    
    group by charttimes.uniquepid, charttimes.patienthealthsystemstayid
)

select
    pivot1.uniquepid, 
    pivot1.patienthealthsystemstayid,
    pivot1.admission_cause,
    pivot1.surgery,
    pivot1.elective_surgery,
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
    on pivot2.patienthealthsystemstayid = pivot1.patienthealthsystemstayid
    
);
