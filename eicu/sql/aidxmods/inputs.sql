-- --------------------------------------------------------------------------------------------------------------------
-- Inputs
-- --------------------------------------------------------------------------------------------------------------------
-- Tratamientos como infusión de vasopresores y ventilación mecánica.

-- Parámetro                        UOM                     Rango         
-- -----------------------------------------------------------------------
-- Dopamina                         mcg/kg/min              ]0, -- [
-- Epinefrina                       mcg/kg/min              ]0, -- [
-- Norepinefrina                    mcg/kg/min              ]0, -- [
-- Dobutamina                       mcg/kg/min              ]0, -- [
-- Ventilación                      bool                    bool
-- -----------------------------------------------------------------------


set search_path to public, eicu_crd;

drop table if exists aidxmods.inputs;

create table aidxmods.inputs as (

with pivot1 as (
    select 
        charttimes.patientunitstayid as patientunitstayid,
        charttimes.los as los,
        case
            when drugname = 'Dopamine (mcg/kg/min)'
                and drugrate ~ '^[-]?[0-9]+[.]?[0-9]*$' and drugrate is not null and drugrate != ''
                and drugrate::numeric > 0
                then drugrate::numeric
            when drugname = 'dopamine (mcg/kg/min)'
                and drugrate ~ '^[-]?[0-9]+[.]?[0-9]*$' and drugrate is not null and drugrate != ''
                and drugrate::numeric > 0
                then drugrate::numeric
            when drugname = 'Dopamine (mcg/kg/hr)'
                and drugrate ~ '^[-]?[0-9]+[.]?[0-9]*$' and drugrate is not null and drugrate != ''
                and drugrate::numeric > 0
                then drugrate::numeric / 60
            when drugname = 'Dopamine (nanograms/kg/min)' 
                and drugrate ~ '^[-]?[0-9]+[.]?[0-9]*$' and drugrate is not null and drugrate != ''
                and drugrate::numeric > 0
                then drugrate::numeric / 1000
            when drugname = 'Dopamine (mcg/hr)'
                and drugrate ~ '^[-]?[0-9]+[.]?[0-9]*$' and drugrate is not null and drugrate != ''
                and drugrate::numeric > 0
                then drugrate::numeric / nullif(patient.admissionweight, 0) / 60
            when drugname = 'Dopamine (mcg/min)'
                and drugrate ~ '^[-]?[0-9]+[.]?[0-9]*$' and drugrate is not null and drugrate != ''
                and drugrate::numeric > 0
                then drugrate::numeric / nullif(patient.admissionweight, 0)
            when drugname = 'Dopamine (mg/hr)'
                and drugrate ~ '^[-]?[0-9]+[.]?[0-9]*$' and drugrate is not null and drugrate != ''
                and drugrate::numeric > 0
                then drugrate::numeric * 1000 / nullif(patient.admissionweight, 0) / 60
            when drugname = 'DOPamine STD 15 mg Dextrose 5% 250 ml  Premix (mcg/kg/min)' 
                and drugrate ~ '^[-]?[0-9]+[.]?[0-9]*$' and drugrate is not null and drugrate != ''
                and drugrate::numeric > 0
                then drugrate::numeric
            when drugname = 'DOPamine STD 400 mg Dextrose 5% 250 ml  Premix (mcg/kg/min)'
                and drugrate ~ '^[-]?[0-9]+[.]?[0-9]*$' and drugrate is not null and drugrate != ''
                and drugrate::numeric > 0
                then drugrate::numeric
            when drugname = 'DOPamine STD 400 mg Dextrose 5% 500 ml  Premix (mcg/kg/min)'
                and drugrate ~ '^[-]?[0-9]+[.]?[0-9]*$' and drugrate is not null and drugrate != ''
                and drugrate::numeric > 0
                then drugrate::numeric
            when drugname = 'DOPamine MAX 800 mg Dextrose 5% 250 ml  Premix (mcg/kg/min)'
                and drugrate ~ '^[-]?[0-9]+[.]?[0-9]*$' and drugrate is not null and drugrate != ''
                and drugrate::numeric > 0
                then drugrate::numeric
            when drugname = 'Dopamine (ml/hr)'
                and drugamount ~ '^[-]?[0-9]+[.]?[0-9]*$' and drugamount is not null and drugamount != ''
                and volumeoffluid ~ '^[-]?[0-9]+[.]?[0-9]*$' and volumeoffluid is not null and volumeoffluid != ''
                and drugrate::numeric > 0
                then drugrate::numeric * (drugamount::numeric / nullif(volumeoffluid::numeric, 0)) * 1000 / nullif(patient.admissionweight, 0) / 60
            else null end
        as dopamine,
        case
            when drugname = 'Epinephrine (mcg/kg/min)'
                and drugrate ~ '^[-]?[0-9]+[.]?[0-9]*$' and drugrate is not null and drugrate != ''
                and drugrate::numeric > 0
                then drugrate::numeric
            when drugname = 'Epinephrine (mg/kg/min)'
                and drugrate ~ '^[-]?[0-9]+[.]?[0-9]*$' and drugrate is not null and drugrate != ''
                and drugrate::numeric > 0
                then drugrate::numeric * 1000
            when drugname = 'Epinepherine (mcg/min)'
                and drugrate ~ '^[-]?[0-9]+[.]?[0-9]*$' and drugrate is not null and drugrate != ''
                and drugrate::numeric > 0
                then drugrate::numeric / nullif(patient.admissionweight, 0)
            when drugname = 'Epinephrine (mcg/min)'
                and drugrate ~ '^[-]?[0-9]+[.]?[0-9]*$' and drugrate is not null and drugrate != ''
                and drugrate::numeric > 0
                then drugrate::numeric / nullif(patient.admissionweight, 0)
            when drugname = 'EPI (mcg/min)'
                and drugrate ~ '^[-]?[0-9]+[.]?[0-9]*$' and drugrate is not null and drugrate != ''
                and drugrate::numeric > 0
                then drugrate::numeric / nullif(patient.admissionweight, 0)
            when drugname = 'Epinephrine (mcg/hr)'
                and drugrate ~ '^[-]?[0-9]+[.]?[0-9]*$' and drugrate is not null and drugrate != ''
                and drugrate::numeric > 0
                then drugrate::numeric / nullif(patient.admissionweight, 0) / 60
            when drugname = 'Epinephrine (mg/hr)'
                and drugrate ~ '^[-]?[0-9]+[.]?[0-9]*$' and drugrate is not null and drugrate != ''
                and drugrate::numeric > 0
                then drugrate::numeric * 1000  / nullif(patient.admissionweight, 0) / 60
            when drugname = 'EPINEPHrine(Adrenalin)MAX 30 mg Sodium Chloride 0.9% 250 ml (mcg/min)'
                and drugrate ~ '^[-]?[0-9]+[.]?[0-9]*$' and drugrate is not null and drugrate != ''
                and drugrate::numeric > 0
                then drugrate::numeric / nullif(patient.admissionweight, 0)
            when drugname = 'EPINEPHrine(Adrenalin)STD 4 mg Sodium Chloride 0.9% 250 ml (mcg/min)'
                and drugrate ~ '^[-]?[0-9]+[.]?[0-9]*$' and drugrate is not null and drugrate != ''
                and drugrate::numeric > 0
                then drugrate::numeric / nullif(patient.admissionweight, 0)
            when drugname = 'EPINEPHrine(Adrenalin)STD 4 mg Sodium Chloride 0.9% 500 ml (mcg/min)'
                and drugrate ~ '^[-]?[0-9]+[.]?[0-9]*$' and drugrate is not null and drugrate != ''
                and drugrate::numeric > 0
                then drugrate::numeric / nullif(patient.admissionweight, 0)
            when drugname = 'EPINEPHrine(Adrenalin)STD 7 mg Sodium Chloride 0.9% 250 ml (mcg/min)'
                and drugrate ~ '^[-]?[0-9]+[.]?[0-9]*$' and drugrate is not null and drugrate != ''
                and drugrate::numeric > 0
                then drugrate::numeric / nullif(patient.admissionweight, 0)
            when drugname = 'Epinephrine (ml/hr)'
                and drugamount ~ '^[-]?[0-9]+[.]?[0-9]*$' and drugamount is not null and drugamount != ''
                and volumeoffluid ~ '^[-]?[0-9]+[.]?[0-9]*$' and volumeoffluid is not null and volumeoffluid != ''
                and drugrate::numeric > 0
                then drugrate::numeric * (drugamount::numeric / nullif(volumeoffluid::numeric, 0)) * 1000 / nullif(patient.admissionweight, 0) / 60
            else null end
        as epinephrine,
        case
            when drugname = 'Norepinephrine (mcg/kg/min)'
                and drugrate ~ '^[-]?[0-9]+[.]?[0-9]*$' and drugrate is not null and drugrate != ''
                and drugrate::numeric > 0
                then drugrate::numeric
            when drugname = 'Norepinephrine (mcg/kg/hr)'
                and drugrate ~ '^[-]?[0-9]+[.]?[0-9]*$' and drugrate is not null and drugrate != ''
                and drugrate::numeric > 0
                then drugrate::numeric / 60
            when drugname = 'Norepinephrine (mg/kg/min)'
                and drugrate ~ '^[-]?[0-9]+[.]?[0-9]*$' and drugrate is not null and drugrate != ''
                and drugrate::numeric > 0
                then drugrate::numeric * 1000
            when drugname = 'Norepinephrine (mcg/min)'
                and drugrate ~ '^[-]?[0-9]+[.]?[0-9]*$' and drugrate is not null and drugrate != ''
                and drugrate::numeric > 0
                then drugrate::numeric / nullif(patient.admissionweight, 0)
            when drugname = 'Norepinephrine (mcg/hr)'
                and drugrate ~ '^[-]?[0-9]+[.]?[0-9]*$' and drugrate is not null and drugrate != ''
                and drugrate::numeric > 0
                then drugrate::numeric / nullif(patient.admissionweight, 0) / 60
            when drugname = 'Norepinephrine (mg/min)'
                and drugrate ~ '^[-]?[0-9]+[.]?[0-9]*$' and drugrate is not null and drugrate != ''
                and drugrate::numeric > 0
                then drugrate::numeric * 1000 / nullif(patient.admissionweight, 0)
            when drugname = 'Norepinephrine (mg/hr)'
                and drugrate ~ '^[-]?[0-9]+[.]?[0-9]*$' and drugrate is not null and drugrate != ''
                and drugrate::numeric > 0
                then drugrate::numeric * 1000  / nullif(patient.admissionweight, 0) / 60
            when drugname = 'Norepinephrine MAX 32 mg Dextrose 5% 250 ml (mcg/min)'
                and drugrate ~ '^[-]?[0-9]+[.]?[0-9]*$' and drugrate is not null and drugrate != ''
                and drugrate::numeric > 0
                then drugrate::numeric / nullif(patient.admissionweight, 0)
            when drugname = 'Norepinephrine MAX 32 mg Dextrose 5% 500 ml (mcg/min)'
                and drugrate ~ '^[-]?[0-9]+[.]?[0-9]*$' and drugrate is not null and drugrate != ''
                and drugrate::numeric > 0
                then drugrate::numeric / nullif(patient.admissionweight, 0)
            when drugname = 'Norepinephrine STD 32 mg Dextrose 5% 282 ml (mcg/min)'
                and drugrate ~ '^[-]?[0-9]+[.]?[0-9]*$' and drugrate is not null and drugrate != ''
                and drugrate::numeric > 0
                then drugrate::numeric / nullif(patient.admissionweight, 0)
            when drugname = 'Norepinephrine STD 32 mg Dextrose 5% 500 ml (mcg/min)'
                and drugrate ~ '^[-]?[0-9]+[.]?[0-9]*$' and drugrate is not null and drugrate != ''
                and drugrate::numeric > 0
                then drugrate::numeric / nullif(patient.admissionweight, 0)
            when drugname = 'Norepinephrine STD 4 mg Dextrose 5% 250 ml (mcg/min)'
                and drugrate ~ '^[-]?[0-9]+[.]?[0-9]*$' and drugrate is not null and drugrate != ''
                and drugrate::numeric > 0
                then drugrate::numeric / nullif(patient.admissionweight, 0)
            when drugname = 'Norepinephrine STD 4 mg Dextrose 5% 500 ml (mcg/min)'
                and drugrate ~ '^[-]?[0-9]+[.]?[0-9]*$' and drugrate is not null and drugrate != ''
                and drugrate::numeric > 0
                then drugrate::numeric / nullif(patient.admissionweight, 0)
            when drugname = 'Norepinephrine STD 8 mg Dextrose 5% 250 ml (mcg/min)'
                and drugrate ~ '^[-]?[0-9]+[.]?[0-9]*$' and drugrate is not null and drugrate != ''
                and drugrate::numeric > 0
                then drugrate::numeric / nullif(patient.admissionweight, 0)
            when drugname = 'Norepinephrine STD 8 mg Dextrose 5% 500 ml (mcg/min)'
                and drugrate ~ '^[-]?[0-9]+[.]?[0-9]*$' and drugrate is not null and drugrate != ''
                and drugrate::numeric > 0
                then drugrate::numeric / nullif(patient.admissionweight, 0)
            when drugname = 'Norepinephrine (ml/hr)'
                and drugamount ~ '^[-]?[0-9]+[.]?[0-9]*$' and drugamount is not null and drugamount != ''
                and volumeoffluid ~ '^[-]?[0-9]+[.]?[0-9]*$' and volumeoffluid is not null and volumeoffluid != ''
                and drugrate::numeric > 0
                then drugrate::numeric * (drugamount::numeric / nullif(volumeoffluid::numeric, 0)) * 1000 / nullif(patient.admissionweight, 0) / 60
            when drugname = 'norepinephrine Volume (ml) (ml/hr)'
                and drugamount ~ '^[-]?[0-9]+[.]?[0-9]*$' and drugamount is not null and drugamount != ''
                and volumeoffluid ~ '^[-]?[0-9]+[.]?[0-9]*$' and volumeoffluid is not null and volumeoffluid != ''
                and drugrate::numeric > 0
                then drugrate::numeric * (drugamount::numeric / nullif(volumeoffluid::numeric, 0)) * 1000 / nullif(patient.admissionweight, 0) / 60
            when drugname = 'Levophed (mcg/kg/min)'
                and drugrate ~ '^[-]?[0-9]+[.]?[0-9]*$' and drugrate is not null and drugrate != ''
                and drugrate::numeric > 0
                then drugrate::numeric
            when drugname = 'levophed  (mcg/min)'
                and drugrate ~ '^[-]?[0-9]+[.]?[0-9]*$' and drugrate is not null and drugrate != ''
                and drugrate::numeric > 0
                then drugrate::numeric / nullif(patient.admissionweight, 0)
            when drugname = 'Levophed (mcg/min)'
                and drugrate ~ '^[-]?[0-9]+[.]?[0-9]*$' and drugrate is not null and drugrate != ''
                and drugrate::numeric > 0
                then drugrate::numeric / nullif(patient.admissionweight, 0)
            when drugname = 'levophed (mcg/min)'
                and drugrate ~ '^[-]?[0-9]+[.]?[0-9]*$' and drugrate is not null and drugrate != ''
                and drugrate::numeric > 0
                then drugrate::numeric / nullif(patient.admissionweight, 0)
            when drugname = 'Levophed (mg/hr)'
                and drugrate ~ '^[-]?[0-9]+[.]?[0-9]*$' and drugrate is not null and drugrate != ''
                and drugrate::numeric > 0
                then drugrate::numeric * 1000 / nullif(patient.admissionweight, 0)
            when drugname = 'Levophed (ml/hr)'
                and drugamount ~ '^[-]?[0-9]+[.]?[0-9]*$' and drugamount is not null and drugamount != ''
                and volumeoffluid ~ '^[-]?[0-9]+[.]?[0-9]*$' and volumeoffluid is not null and volumeoffluid != ''
                and drugrate::numeric > 0
                then drugrate::numeric * (drugamount::numeric / nullif(volumeoffluid::numeric, 0)) * 1000 / nullif(patient.admissionweight, 0) / 60
            when drugname = 'levophed (ml/hr)'
                and drugamount ~ '^[-]?[0-9]+[.]?[0-9]*$' and drugamount is not null and drugamount != ''
                and volumeoffluid ~ '^[-]?[0-9]+[.]?[0-9]*$' and volumeoffluid is not null and volumeoffluid != ''
                and drugrate::numeric > 0
                then drugrate::numeric * (drugamount::numeric / nullif(volumeoffluid::numeric, 0)) * 1000 / nullif(patient.admissionweight, 0) / 60
            when drugname = 'NSS with LEVO (ml/hr)'
                and drugamount ~ '^[-]?[0-9]+[.]?[0-9]*$' and drugamount is not null and drugamount != ''
                and volumeoffluid ~ '^[-]?[0-9]+[.]?[0-9]*$' and volumeoffluid is not null and volumeoffluid != ''
                and drugrate::numeric > 0
                then drugrate::numeric * (drugamount::numeric / nullif(volumeoffluid::numeric, 0)) * 1000 / nullif(patient.admissionweight, 0) / 60
            when drugname = 'NSS w/ levo/vaso (ml/hr)'
                and drugamount ~ '^[-]?[0-9]+[.]?[0-9]*$' and drugamount is not null and drugamount != ''
                and volumeoffluid ~ '^[-]?[0-9]+[.]?[0-9]*$' and volumeoffluid is not null and volumeoffluid != ''
                and drugrate::numeric > 0
                then drugrate::numeric * (drugamount::numeric / nullif(volumeoffluid::numeric, 0)) * 1000 / nullif(patient.admissionweight, 0) / 60
            else null end
        as norepinephrine,
        case
            when drugname = 'Dobutamine (mcg/kg/min)'
                and drugrate ~ '^[-]?[0-9]+[.]?[0-9]*$' and drugrate is not null and drugrate != ''
                and drugrate::numeric > 0
                then drugrate::numeric
            when drugname = 'Dobutamine (mcg/kg/hr)'
                and drugrate ~ '^[-]?[0-9]+[.]?[0-9]*$' and drugrate is not null and drugrate != ''
                and drugrate::numeric > 0
                then drugrate::numeric / 60
            when drugname = 'Dobutamine (mcg/min)'
                and drugrate ~ '^[-]?[0-9]+[.]?[0-9]*$' and drugrate is not null and drugrate != ''
                and drugrate::numeric > 0
                then drugrate::numeric / nullif(patient.admissionweight, 0)
            when drugname = 'DOBUTamine STD 500 mg Dextrose 5% 250 ml  Premix (mcg/kg/min)'
                and drugrate ~ '^[-]?[0-9]+[.]?[0-9]*$' and drugrate is not null and drugrate != ''
                and drugrate::numeric > 0
                then drugrate::numeric
            when drugname = 'DOBUTamine MAX 1000 mg Dextrose 5% 250 ml  Premix (mcg/kg/min)'
                and drugrate ~ '^[-]?[0-9]+[.]?[0-9]*$' and drugrate is not null and drugrate != ''
                and drugrate::numeric > 0
                then drugrate::numeric
            when drugname = 'Dobutamine (ml/hr)'
                and drugamount ~ '^[-]?[0-9]+[.]?[0-9]*$' and drugamount is not null and drugamount != ''
                and volumeoffluid ~ '^[-]?[0-9]+[.]?[0-9]*$' and volumeoffluid is not null and volumeoffluid != ''
                and drugrate::numeric > 0
                then drugrate::numeric * (drugamount::numeric / nullif(volumeoffluid::numeric, 0)) * 1000 / nullif(patient.admissionweight, 0) / 60
            when drugname = 'dobutrex (mcg/kg/min)'
                and drugrate ~ '^[-]?[0-9]+[.]?[0-9]*$' and drugrate is not null and drugrate != ''
                and drugrate::numeric > 0
                then drugrate::numeric
            when drugname = 'dobutrex (mg/kg/min)'
                and drugrate ~ '^[-]?[0-9]+[.]?[0-9]*$' and drugrate is not null and drugrate != ''
                and drugrate::numeric > 0
                then drugrate::numeric
            else null end
        as dobutamine

        from firstday.charttimes charttimes

    left join patient
        on patient.patientunitstayid = charttimes.patientunitstayid
    
    left join infusiondrug
        on infusiondrug.patientunitstayid = charttimes.patientunitstayid
        and infusiondrug.infusionoffset >= charttimes.starttime
        and infusiondrug.infusionoffset <  charttimes.endtime

    where infusiondrug.patientunitstayid is not null
),
    
pivot2 as (
    select
        patientunitstayid,
        charttime,
        case
            when
                string in (
                    'plateau pressure',
                    'postion at lip',
                    'position at lip',
                    'pressure control',
                    'bi-pap',
                    'ambubag',
                    'flowtrigger',
                    'peep',
                    'tv/kg ibw',
                    'mean airway pressure',
                    'peak insp. pressure',
                    'exhaled mv',
                    'exhaled tv (machine)',
                    'exhaled tv (patient)',
                    'flow sensitivity',
                    'peak flow',
                    'f total',
                    'pressure to trigger ps',
                    'adult con setting set rr',
                    'adult con setting set vt',
                    'vti',
                    'exhaled vt',
                    'adult con alarms hi press alarm',
                    'mve',
                    'respiratory phase',
                    'inspiratory pressure, set',
                    'a1: high exhaled vt',
                    'set fraction of inspired oxygen (fio2)',
                    'insp flow (l/min)',
                    'adult con setting spont exp vt',
                    'spont tv',
                    'pulse ox results vt',
                    'vt spontaneous (ml)',
                    'peak pressure',
                    'ltv1200',
                    'tc'
                )
                or (
                    string like '%vent%'
                    and not string like '%hyperventilat%'
                )
                or string like '%set vt%'
                or string like '%sputum%'
                or string like '%rsbi%'
                or string like '%tube%'
                or string like '%ett%'
                or string like '%endotracheal%'
                or string like '%tracheal suctioning%'
                or string like '%tracheostomy%'
                or string like '%reintubation%'
                or string like '%assist controlled%'
                or string like '%volume controlled%'
                or string like '%pressure controlled%'
                or string like '%trach collar%'
                or string like '%ipap%'
                or string like '%niv%'
                or string like '%epap%'
                or string like '%mask leak%'
                or string like '%volume assured%'
                or string like '%non-invasive ventilation%'
                or string like '%cpap%'
                or string like '%tidal%'
                or string like '%flow rate%'
                or string like '%minute volume%'
                or string like '%leak%'
                or string like '%pressure support%'
                or string like '%peep%'
                or string like '%tidal volume%'
            then 1
            else 0
        end as ventilation,
        case
            when string in ('fio2', 'fio2 (%)')
                and value >  0.21
                and value <= 1.00
                then value * 100
            when string in ('fio2', 'fio2 (%)')
                and value >  21
                and value <= 100
                then value
            else null
        end as fio2

    from (
        select patientunitstayid,
            nursingChartOffset as charttime,
            lower(nursingchartvalue) as string,
            null as value
        from nursecharting

        union all

        select patientunitstayid,
            respchartoffset as charttime,
            lower(respchartvaluelabel) as string,
            case
                when lower(respchartvaluelabel) in ('fio2', 'fio2 (%)')
                    then replace(respchartvalue, '%', '')::numeric
                else null
            end as value
        from respiratorycharting

        union all

        select patientunitstayid,
            respchartoffset as charttime,
            LOWER(respchartvalue) AS string,
            null as value
        from respiratorycharting
        where lower(respchartvaluelabel) in (
            'o2 device',
            'respiratory device',
            'ventilator type',
            'oxygen delivery method'
        )
        
        union all
        
        select patientunitstayid,
            labresultoffset as charttime,
            'fio2' as string,
            labresult as value
        from lab
        where lower(labname) = 'fio2'
        
    ) sources
),

pivot3 as (
    select 
        charttimes.patientunitstayid as patientunitstayid,
        charttimes.los as los,
        pivot2.ventilation as ventilation,
        pivot2.fio2 as fio2

    from firstday.charttimes charttimes

    left join pivot2
        on pivot2.patientunitstayid = charttimes.patientunitstayid
        and pivot2.charttime >= charttimes.starttime
        and pivot2.charttime <  charttimes.endtime
)


select
    charttimes.uniquepid as uniquepid,
    charttimes.patienthealthsystemstayid as patienthealthsystemstayid,
    charttimes.patientunitstayid as patientunitstayid,
    charttimes.los as los,
    round(min(pivot1.dopamine), 2) as dopamine_min,
    round(avg(pivot1.dopamine), 2) as dopamine_avg,
    round(max(pivot1.dopamine), 2) as dopamine_max,
    round(min(pivot1.epinephrine), 2) as epinephrine_min,
    round(avg(pivot1.epinephrine), 2) as epinephrine_avg,
    round(max(pivot1.epinephrine), 2) as epinephrine_max,
    round(min(pivot1.norepinephrine), 2) as norepinephrine_min,
    round(avg(pivot1.norepinephrine), 2) as norepinephrine_avg,
    round(max(pivot1.norepinephrine), 2) as norepinephrine_max,
    round(min(pivot1.dobutamine), 2) as dobutamine_min,
    round(avg(pivot1.dobutamine), 2) as dobutamine_avg,
    round(max(pivot1.dobutamine), 2) as dobutamine_max,
    round(min(pivot3.fio2), 2) as fio2_min,
    round(avg(pivot3.fio2), 2) as fio2_avg,
    round(max(pivot3.fio2), 2) as fio2_max,
    max(pivot3.ventilation) as ventilation
    

from firstday.charttimes charttimes
left join pivot1
    on pivot1.patientunitstayid = charttimes.patientunitstayid
left join pivot3
    on pivot3.patientunitstayid = charttimes.patientunitstayid
    
group by charttimes.uniquepid, charttimes.patienthealthsystemstayid, charttimes.patientunitstayid, charttimes.los
    
);
