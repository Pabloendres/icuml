-- --------------------------------------------------------------------------------------------------------------------
-- GCS
-- --------------------------------------------------------------------------------------------------------------------
-- Glasgow Coma Scale.

-- Parámetro                        UOM                     Rango         
-- -----------------------------------------------------------------------
-- GCS motor                        --                      [1, 6]
-- GCS verbal                       --                      [1, 5]
-- GCS ocular                       --                      [1, 4]
-- GCS total                        --                      [3, 15]
-- -----------------------------------------------------------------------


set search_path to public, eicu_crd;

drop table if exists aidxmods.gcs;

create table aidxmods.gcs as (

with pivot1 as (
    select 
        charttimes.uniquepid as uniquepid,
        charttimes.patienthealthsystemstayid as patienthealthsystemstayid,
        charttimes.patientunitstayid as patientunitstayid,
        charttimes.los as los,
        case
            when nursingchartcelltypevallabel = 'Glasgow coma score'
                and nursingchartcelltypevalname = 'Motor'
                and nursingchartvalue ~ '^[1-6]+[.]?[0-9]*$'
                and nursingchartvalue not in ('-','.')
                then nursingchartvalue::int
            when nursingchartcelltypevallabel in ('Best Motor Response', 'Motor Response')
                and nursingchartvalue in ('1', '1-->(M1) none', 'Flaccid')
                then 1
            when nursingchartcelltypevallabel in ('Best Motor Response', 'Motor Response')
                and nursingchartvalue in ('2', '2-->(M2) extension to pain', 'Abnormal extension')
                then 2
            when nursingchartcelltypevallabel in ('Best Motor Response', 'Motor Response')
                and nursingchartvalue in ('3', '3-->(M3) flexion to pain', 'Abnormal flexion')
                then 3
            when nursingchartcelltypevallabel in ('Best Motor Response', 'Motor Response')
                and nursingchartvalue in ('4', '4-->(M4) withdraws from pain', 'Withdraws') 
                then 4
            when nursingchartcelltypevallabel in ('Best Motor Response', 'Motor Response')
                and nursingchartvalue in ('5', '5-->(M5) localizes pain', 'Localizes to noxious stimuli')
                then 5
            when nursingchartcelltypevallabel in ('Best Motor Response', 'Motor Response')
                and nursingchartvalue in ('6','6-->(M6) obeys commands', 'Obeys simple commands')
                then 6
            else null end
        as motor,
        case
            when nursingchartcelltypevallabel = 'Glasgow coma score'
                and nursingchartcelltypevalname = 'Verbal'
                and nursingchartvalue ~ '^[1-5]+[.]?[0-9]*$'
                and nursingchartvalue not in ('-','.')
                then nursingchartvalue::int
            when nursingchartcelltypevallabel in ('Best Verbal Response', 'Verbal Response')
                and nursingchartvalue in ('1', '1-->(V1) none', 'None', 'Clearly unresponsive')
                then 1
            when nursingchartcelltypevallabel in ('Best Verbal Response', 'Verbal Response')
                and nursingchartvalue in ('2', '2-->(V2) incomprehensible speech', 'Incomprehensible sounds')
                then 2
            when nursingchartcelltypevallabel in ('Best Verbal Response', 'Verbal Response')
                and nursingchartvalue in ('3', '3-->(V3) inappropriate words', 'Inappropriate words')
                then 3
            when nursingchartcelltypevallabel in ('Best Verbal Response', 'Verbal Response')
                and nursingchartvalue in ('4', '4-->(V4) confused', 'Confused')
                then 4
            when nursingchartcelltypevallabel in ('Best Verbal Response', 'Verbal Response')
                and nursingchartvalue in ('5', '5-->(V5) oriented', 'Oriented', 'Orientation/ability to communicate questionable', 'Clearly oriented/can indicate needs')
                then 5
            else null end
        as verbal,
        case
            when nursingchartcelltypevallabel = 'Glasgow coma score'
                and nursingchartcelltypevalname = 'Eyes'
                and nursingchartvalue ~ '^[1-4]+[.]?[0-9]*$'
                and nursingchartvalue not in ('-','.')
                then nursingchartvalue::int
            when nursingchartcelltypevallabel in ('Best Eye Response', 'Eye Opening')
                and nursingchartvalue in ('1', '1-->(E1) none')
                then 1
            when nursingchartcelltypevallabel in ('Best Eye Response', 'Eye Opening')
                and nursingchartvalue in ('2', '2-->(E2) to pain')
                then 2
            when nursingchartcelltypevallabel in ('Best Eye Response', 'Eye Opening')
                and nursingchartvalue in ('3', '3-->(E3) to speech')
                then 3
            when nursingchartcelltypevallabel in ('Best Eye Response', 'Eye Opening')
                and nursingchartvalue in ('4', '4-->(E4) spontaneous')
                then 4
            else null end
       as eyes,
       case
            when nursingchartcelltypevallabel in ('Best Verbal Response', 'Verbal Response')
                and nursingchartvalue = 'Trached or intubated'
                then null
            when nursingchartcelltypevallabel = 'Glasgow coma score'
                and nursingchartcelltypevalname = 'GCS Total'
                and nursingchartvalue = 'Unable to score due to medication'
                then null
            when nursingchartcelltypevallabel = 'Glasgow coma score'
                and nursingchartcelltypevalname = 'GCS Total'
                and nursingchartvalue ~ '^[1-9]+[.]?[0-9]*$'
                and nursingchartvalue not in ('-','.')
                and nursingchartvalue::int between 3 and 15
                then nursingchartvalue::int
            when nursingchartcelltypevallabel = 'Score (Glasgow Coma Scale)'
                and nursingchartcelltypevalname = 'Value'
                and nursingchartvalue ~ '^[1-9]+[.]?[0-9]*$'
                and nursingchartvalue not in ('-','.')
                and nursingchartvalue::int between 3 and 15
                then nursingchartvalue::int
            else null end
        as gcs

    from firstday.charttimes charttimes

    left join nursecharting
        on nursecharting.patientunitstayid = charttimes.patientunitstayid
        and nursecharting.nursingchartoffset >= charttimes.starttime 
        and nursecharting.nursingchartoffset <  charttimes.endtime
        and nursecharting.nursingchartcelltypecat in (
            'Scores', 'Other Vital Signs and Infusions'
        )
),

pivot2 as (
    select 
        uniquepid,
        patienthealthsystemstayid,
        patientunitstayid,
        los,
        min(verbal)::int as verbal_min,
        avg(verbal)::int as verbal_avg,
        max(verbal)::int as verbal_max,
        min(motor)::int as motor_min,
        avg(motor)::int as motor_avg,
        max(motor)::int as motor_max,
        min(eyes)::int as eyes_min,
        avg(eyes)::int as eyes_avg,
        max(eyes)::int as eyes_max

    from pivot1

    group by pivot1.uniquepid, pivot1.patienthealthsystemstayid, pivot1.patientunitstayid, pivot1.los
)
    
select
    *,
    (verbal_min + motor_min + eyes_min)::int as gcs_min,
    (verbal_avg + motor_avg + eyes_avg)::int as gcs_avg,
    (verbal_max + motor_max + eyes_max)::int as gcs_max

from pivot2

);
