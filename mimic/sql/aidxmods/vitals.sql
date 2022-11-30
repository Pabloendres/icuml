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
-- Fracción inspirada de oxígeno    %                       [21, 100]
-- BNP                              pg/mL                   ]0, -- [
-- Fibrinógeno                      mg/dL                   ]0, -- [
-- -----------------------------------------------------------------------


set search_path to public, mimiciii;

drop table if exists aidxmods.vitals;

create table aidxmods.vitals as (

with pivot as (
    select 
        charttimes.subject_id as subject_id,
        charttimes.hadm_id as hadm_id, 
        charttimes.icustay_id as icustay_id,
        charttimes.los as los,
        case
            when itemid in (211, 220045)
                and valuenum::numeric >  0
                and valuenum::numeric <  400
                then valuenum::numeric
            else null end
        as heartrate,
        case
            when itemid in (615, 618, 220210, 224690)
                and valuenum::numeric >  0
                and valuenum::numeric <  120
                then valuenum::numeric
            else null end
        as resprate,
        case
            when itemid in (223762, 676)
                and valuenum::numeric >  32
                and valuenum::numeric <  45
                then valuenum::numeric
            when itemid in (223761, 678)
                and (valuenum::numeric - 32) / 1.8 >  32
                and (valuenum::numeric - 32) / 1.8 <  45
                then (valuenum::numeric - 32) / 1.8 -- Conv. °F to °C
            else null end
        as temperature,
        case
            when itemid in (51, 442, 455, 6701, 220179, 220050)
                and valuenum::numeric >  0
                and valuenum::numeric <  300
                then valuenum::numeric
            else null end
        as sbp,
        case
            when itemid in (8368, 8440, 8441, 8555, 220180, 220051)
                and valuenum::numeric >  0
                and valuenum::numeric <  300
                then valuenum::numeric
            else null end
        as dbp,
        case
            when itemid in (456, 52, 6702, 443, 220052, 220181, 225312)
                and valuenum::numeric >  0
                and valuenum::numeric <  300
                then valuenum::numeric
            else null end
        as map,
        case
            when itemid in (113, 220074)
                and valuenum::numeric >  0
                and valuenum::numeric <  300
                then valuenum::numeric
            else null end
        as cvp,
        case
            when itemid in (646, 220277)
                and valuenum::numeric >  0
                and valuenum::numeric <= 100
                then valuenum::numeric
            else null end
        as spo2,
        case
            when itemid in (190, 3420, 3422, 223835)
                and valuenum::numeric >= 21
                and valuenum::numeric <= 100
                then valuenum::numeric
             when itemid in (190, 3420, 3422, 223835)
                and valuenum::numeric >= 0.21
                and valuenum::numeric <= 1.00
                then valuenum::numeric * 100 -- Conv. Fraction to Percent
            else null end
        as fio2,
        case
            when itemid in (7294, 227446, 225622)
                and valuenum::numeric >  0
                then valuenum::numeric
            else null end
        as bnp,
        case
            when itemid in (1528, 227468, 806, 220541)
                and valuenum::numeric >  0
                then valuenum::numeric
            else null end
        as fibrinogen

    from firstday.charttimes charttimes

    left join chartevents
        on chartevents.icustay_id = charttimes.icustay_id
        and (chartevents.error is NULL or chartevents.error = 0)
        and chartevents.charttime >= charttimes.starttime
        and chartevents.charttime <  charttimes.endtime
        and chartevents.itemid in (
            -- comment is: LABEL | DBSOURCE
            211,    -- Arterial BP [Systolic]                | carevue
            220045, -- Arterial BP Mean                      | carevue
            51,     -- Heart Rate                            | carevue
            442,    -- Manual BP [Systolic]                  | carevue
            455,    -- Manual BP Mean(calc)                  | carevue
            6701,   -- NBP [Systolic]                        | carevue
            220179, -- NBP Mean                              | carevue
            220050, -- Resp Rate (Total)                     | carevue
            8368,   -- Respiratory Rate                      | carevue
            8440,   -- SpO2                                  | carevue
            8441,   -- Temperature C                         | carevue
            8555,   -- Temperature F                         | carevue
            220180, -- Fingerstick Glucose                   | carevue
            220051, -- Glucose (70-105)                      | carevue
            456,    -- Glucose                               | carevue
            52,     -- Blood Glucose                         | carevue
            6702,   -- BloodGlucose                          | carevue
            443,    -- Arterial BP #2 [Systolic]             | carevue
            220052, -- Arterial BP Mean #2                   | carevue
            220181, -- Arterial BP [dbp]                     | carevue
            225312, -- Manual BP [Diastolic]                 | carevue
            618,    -- NBP [Diastolic]                       | carevue
            615,    -- Arterial BP #2 [Diastolic]            | carevue
            190,    -- FiO2 Set                              | carevue
            3420,   -- FIO2                                  | carevue
            3422,   -- FIO2 [Meas]                           | carevue
            113,    -- CVP                                   | carevue
            7294,   -- BNP                                   | carevue
            1528,   -- Fibrinogen                            | carevue
            806,    -- Fibrinogen (150-400)                  | carevue
            220210, -- Heart Rate                            | metavision
            224690, -- Arterial Blood Pressure systolic      | metavision
            646,    -- Arterial Blood Pressure diastolic     | metavision
            220277, -- Arterial Blood Pressure mean          | metavision
            807,    -- Non Invasive Blood Pressure systolic  | metavision
            811,    -- Non Invasive Blood Pressure diastolic | metavision
            1529,   -- Non Invasive Blood Pressure mean      | metavision
            3745,   -- Respiratory Rate                      | metavision
            3744,   -- O2 saturation pulseoxymetry           | metavision
            225664, -- Glucose (serum)                       | metavision
            220621, -- Temperature Fahrenheit                | metavision
            226537, -- Temperature Celsius                   | metavision
            223762, -- Respiratory Rate (Total)              | metavision
            676,    -- ART BP mean                           | metavision
            223761, -- Glucose finger stick                  | metavision
            678,    -- Glucose (whole blood)                 | metavision
            223835, -- Inspired O2 Fraction                  | metavision
            220074, -- CVP                                   | metavision
            227446, -- Brain Natiuretic Peptide (BNP)        | metavision
            225622, -- ZBrain Natiuretic Peptide (BNP)       | metavision
            227468, -- Fibrinogen                            | metavision
            220541 -- ZFibrinogen                           | metavision
    )
)

select
    subject_id,
    hadm_id, 
    icustay_id,
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
    round(max(spo2), 2) as spo2_max,
    round(min(fio2), 2) as fio2_min,
    round(avg(fio2), 2) as fio2_avg,
    round(max(fio2), 2) as fio2_max,
    round(min(bnp), 2) as bnp_min,
    round(avg(bnp), 2) as bnp_avg,
    round(max(bnp), 2) as bnp_max,
    round(min(fibrinogen), 2) as fibrinogen_min,
    round(avg(fibrinogen), 2) as fibrinogen_avg,
    round(max(fibrinogen), 2) as fibrinogen_max
    

from pivot

group by pivot.subject_id, pivot.hadm_id, pivot.icustay_id, pivot.los
order by pivot.subject_id, pivot.hadm_id, pivot.icustay_id, pivot.los
    
);
