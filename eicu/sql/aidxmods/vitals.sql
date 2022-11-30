-- --------------------------------------------------------------------------------------------------------------------
-- Vitals
-- --------------------------------------------------------------------------------------------------------------------
-- Signos vitales.

-- Parámetro                        UOM                     Rango         
-- -----------------------------------------------------------------------
-- Ritmo cardiaco                   BPM                     ]0, 400[
-- Ritmo respiratorio               RPM                     ]0, 120[
-- Temperatura                      °C                      [25, 45[
-- Presión sistólica                mmHg                    ]0, 300[
-- Presión diastólica               mmHg                    ]0, 300[
-- Presión arterial media           mmHg                    ]0, 300[
-- Presión venosa central           mmHg                    ]0, 30[
-- Saturación de O2                 %                       ]0, 100]
-- -----------------------------------------------------------------------


set search_path to public, eicu_crd;

drop table if exists aidxmods.vitals;

create table aidxmods.vitals as (

with pivot1 as (
    select
        charttimes.uniquepid as uniquepid,
        charttimes.patienthealthsystemstayid as patienthealthsystemstayid,
        charttimes.patientunitstayid as patientunitstayid,
        charttimes.los as los,
        case
            when nursingchartcelltypevallabel = 'Heart Rate'
                and nursingchartcelltypevalname = 'Heart Rate'
                and nursingchartvalue ~ '^[-]?[0-9]+[.]?[0-9]*$'
                and nursingchartvalue not in ('-','.')
                and nursingchartvalue::numeric >  0
                and nursingchartvalue::numeric <  400
                then nursingchartvalue::numeric
            else null end
        as heartrate,
        case
            when nursingchartcelltypevallabel = 'Respiratory Rate'
                and nursingchartcelltypevalname = 'Respiratory Rate'
                and nursingchartvalue ~ '^[-]?[0-9]+[.]?[0-9]*$'
                and nursingchartvalue not in ('-','.')
                and nursingchartvalue::numeric >  0
                and nursingchartvalue::numeric <  120
                then nursingchartvalue::numeric
            else null end
        as resprate,
        case
            when nursingchartcelltypevallabel = 'Temperature'
                and nursingchartcelltypevalname = 'Temperature (C)'
                and nursingchartvalue ~ '^[-]?[0-9]+[.]?[0-9]*$'
                and nursingchartvalue not in ('-','.')
                and nursingchartvalue::numeric >  32
                and nursingchartvalue::numeric <  45
                then nursingchartvalue::numeric
            when nursingchartcelltypevallabel = 'Temperature'
                and nursingchartcelltypevalname = 'Temperature (C)'
                and nursingchartvalue ~ '^[-]?[0-9]+[.]?[0-9]*$'
                and nursingchartvalue not in ('-','.')
                and (nursingchartvalue::numeric - 32) * 5 / 9 >  32
                and (nursingchartvalue::numeric - 32) * 5 / 9 <  45
                then (nursingchartvalue::numeric - 32) * 5 / 9 -- Conv. °F a °C
            else null end
        as temperature,
        case
            when nursingchartcelltypevallabel = 'Non-Invasive BP'
                and nursingchartcelltypevalname = 'Non-Invasive BP Systolic'
                and nursingchartvalue ~ '^[-]?[0-9]+[.]?[0-9]*$'
                and nursingchartvalue not in ('-','.')
                and nursingchartvalue::numeric >  0
                and nursingchartvalue::numeric <  300
                then nursingchartvalue::numeric
            else null end
        as sbp,
        case
            when nursingchartcelltypevallabel = 'Non-Invasive BP'
                and nursingchartcelltypevalname = 'Non-Invasive BP Diastolic'
                and nursingchartvalue ~ '^[-]?[0-9]+[.]?[0-9]*$'
                and nursingchartvalue not in ('-','.')
                and nursingchartvalue::numeric >  0
                and nursingchartvalue::numeric <  300
                then nursingchartvalue::numeric
            else null end
        as dbp,
        case
            when nursingchartcelltypevallabel = 'Invasive BP'
                and nursingchartcelltypevalname = 'Invasive BP Mean'
                and nursingchartvalue ~ '^[-]?[0-9]+[.]?[0-9]*$'
                and nursingchartvalue not in ('-','.')
                and nursingchartvalue::numeric >  0
                and nursingchartvalue::numeric <  300
                then nursingchartvalue::numeric
            when nursingchartcelltypevallabel = 'MAP (mmHg)'
                and nursingchartcelltypevalname = 'Value'
                and nursingchartvalue ~ '^[-]?[0-9]+[.]?[0-9]*$'
                and nursingchartvalue not in ('-','.')
                and nursingchartvalue::numeric >  0
                and nursingchartvalue::numeric <  300
                then nursingchartvalue::numeric
            when nursingchartcelltypevallabel = 'Arterial Line MAP (mmHg)'
                and nursingchartcelltypevalname = 'Value'
                and nursingchartvalue ~ '^[-]?[0-9]+[.]?[0-9]*$'
                and nursingchartvalue not in ('-','.')
                and nursingchartvalue::numeric >  0
                and nursingchartvalue::numeric <  300
                then nursingchartvalue::numeric
            when nursingchartcelltypevallabel = 'PA'
                and  nursingchartcelltypevalname = 'PA Mean'
                and nursingchartvalue ~ '^[-]?[0-9]+[.]?[0-9]*$'
                and nursingchartvalue not in ('-','.')
                and nursingchartvalue::numeric >  0
                and nursingchartvalue::numeric <  300
                then nursingchartvalue::numeric
            else null end
        as map,
        case
            when nursingchartcelltypevallabel = 'CVP'
                and nursingchartcelltypevalname = 'CVP'
                and nursingchartvalue ~ '^[-]?[0-9]+[.]?[0-9]*$'
                and nursingchartvalue not in ('-','.')
                and nursingchartvalue::numeric >  0
                and nursingchartvalue::numeric <  30
                then nursingchartvalue::numeric
            when nursingchartcelltypevallabel = 'CVP (mmHg)'
                and nursingchartcelltypevalname = 'Value'
                and nursingchartvalue ~ '^[-]?[0-9]+[.]?[0-9]*$'
                and nursingchartvalue not in ('-','.')
                and nursingchartvalue::numeric >  0
                and nursingchartvalue::numeric <  30
                then nursingchartvalue::numeric
            else null end
        as cvp,
        case
            when nursingchartcelltypevallabel = 'O2 Saturation'
                and nursingchartcelltypevalname = 'O2 Saturation'
                and nursingchartvalue ~ '^[-]?[0-9]+[.]?[0-9]*$'
                and nursingchartvalue not in ('-','.')
                and nursingchartvalue::numeric >  0
                and nursingchartvalue::numeric <= 100
                then nursingchartvalue::numeric
            else null end
        as spo2

    from firstday.charttimes charttimes

    left join nursecharting
        on nursecharting.patientunitstayid = charttimes.patientunitstayid
        and nursecharting.nursingchartoffset >= charttimes.starttime
        and nursecharting.nursingchartoffset <  charttimes.endtime
        and nursecharting.nursingchartcelltypecat in (
            'Vital Signs', 'Scores', 'Other Vital Signs and Infusions'
        )
),

pivot2 as (
    select
        charttimes.uniquepid as uniquepid,
        charttimes.patienthealthsystemstayid as patienthealthsystemstayid,
        charttimes.patientunitstayid as patientunitstayid,
        charttimes.los as los,
        case
            when vitalperiodic.heartrate is not null
                and vitalperiodic.heartrate >= 10
                and vitalperiodic.heartrate <  300
                then vitalperiodic.heartrate
            else null end
        as heartrate,
        case
            when vitalperiodic.respiration is not null
                and vitalperiodic.respiration >= 10
                and vitalperiodic.respiration <  70
                then vitalperiodic.respiration
            else null end
        as resprate,
        case
            when vitalperiodic.temperature is not null
                and vitalperiodic.temperature >  0
                and vitalperiodic.temperature <  50
                then vitalperiodic.temperature
            when vitalperiodic.temperature is not null
                and vitalperiodic.temperature >= 50
                and vitalperiodic.temperature <  130
                then (vitalperiodic.temperature - 32) * 5 / 9 -- Conv. °F a °C
            else null end
        as temperature,
        case
            when vitalperiodic.systemicSystolic is not null
                and vitalperiodic.systemicSystolic >= 30
                and vitalperiodic.systemicSystolic <  300
                then vitalperiodic.systemicSystolic
            else null end
        as sbp,
        case
            when vitalperiodic.systemicDiastolic is not null
                and vitalperiodic.systemicDiastolic >  0
                and vitalperiodic.systemicDiastolic <  300
                then vitalperiodic.systemicDiastolic
            else null end
        as dbp,
        case
            when vitalperiodic.pamean is not null
                and vitalperiodic.pamean >  0
                and vitalperiodic.pamean <  300
                then vitalperiodic.pamean
            else null end
        as map,
        case
            when vitalperiodic.cvp is not null
                and vitalperiodic.cvp >  0
                and vitalperiodic.cvp <  30
                then vitalperiodic.cvp
            else null end
        as cvp,
        case
            when vitalperiodic.sao2 is not null
                and vitalperiodic.sao2 >  0
                and vitalperiodic.sao2 <= 100
                then vitalperiodic.sao2
            else null end
        as spo2

    from firstday.charttimes charttimes

    left join vitalperiodic
        on vitalperiodic.patientunitstayid = charttimes.patientunitstayid
        and vitalperiodic.observationoffset >= charttimes.starttime 
        and vitalperiodic.observationoffset <  charttimes.endtime
)

select
    uniquepid,
    patienthealthsystemstayid,
    patientunitstayid,
    los,
    round(min(heartrate), 2) as heartrate_min,
    round(avg(heartrate), 2) as heartrate_avg,
    round(max(heartrate), 2) as heartrate_max,
    round(min(resprate), 2) as resprate_min,
    round(avg(resprate), 2) as resprate_avg,
    round(max(resprate), 2) as resprate_max,
    round(min(temperature), 2) as temperature_min,
    round(avg(temperature), 2) as temperature_avg,
    round(max(temperature), 2) as temperature_max,
    round(min(sbp), 2) as sbp_min,
    round(avg(sbp), 2) as sbp_avg,
    round(max(sbp), 2) as sbp_max,
    round(min(dbp), 2) as dbp_min,
    round(avg(dbp), 2) as dbp_avg,
    round(max(dbp), 2) as dbp_max,
    round(min(map), 2) as map_min,
    round(avg(map), 2) as map_avg,
    round(max(map), 2) as map_max,
    round(min(cvp), 2) as cvp_min,
    round(avg(cvp), 2) as cvp_avg,
    round(max(cvp), 2) as cvp_max,
    round(min(spo2), 2) as spo2_min,
    round(avg(spo2), 2) as spo2_avg,
    round(max(spo2), 2) as spo2_max

from (select * from pivot1 union select * from pivot2) as pivot

group by pivot.uniquepid, pivot.patienthealthsystemstayid, pivot.patientunitstayid, pivot.los
    
);
