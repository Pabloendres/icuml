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

set search_path to public, mimiciii;

drop table if exists aidxmods.gcs;

create table aidxmods.gcs as (

with pivot1 as (
    select 
        charttimes.subject_id as subject_id,
        charttimes.hadm_id as hadm_id, 
        charttimes.icustay_id as icustay_id,
        charttimes.los as los,
        case
            when itemid in (454,223901) --motor
                and valuenum::int >= 1
                and valuenum::int <= 6
                then valuenum::int
            else null end
        as motor,
        case
            when itemid in (723,223900) --verbal
                and value not in ('1.0 ET/Trach', 'No Response-ETT')
                and valuenum::int >= 1
                and valuenum::int <= 5
                then valuenum::int
            else null end
        as verbal,
        case
            when itemid in (184,220739) --eyes
                and valuenum::int >= 1
                and valuenum::int <= 4
                then valuenum::int
            else null end
        as eyes

    from firstday.charttimes charttimes

    left join chartevents
        on chartevents.icustay_id = charttimes.icustay_id
        and (chartevents.error is NULL or chartevents.error = 0)
        and chartevents.charttime >= charttimes.starttime
        and chartevents.charttime <  charttimes.endtime
        and chartevents.itemid in (
            184, 454, 723, 223900, 223901, 220739
        )
),

pivot2 as (
    select 
        subject_id,
        hadm_id, 
        icustay_id,
        los,
        min(motor)::int as motor_min,
        avg(motor)::int as motor_avg,
        max(motor)::int as motor_max,
        min(verbal)::int as verbal_min,
        avg(verbal)::int as verbal_avg,
        max(verbal)::int as verbal_max,
        min(eyes)::int as eyes_min,
        avg(eyes)::int as eyes_avg,
        max(eyes)::int as eyes_max

    from pivot1

    group by pivot1.subject_id, pivot1.hadm_id, pivot1.icustay_id, pivot1.los
    order by pivot1.subject_id, pivot1.hadm_id, pivot1.icustay_id, pivot1.los
)    
    
select
    *,
    (verbal_min + motor_min + eyes_min)::int as gcs_min,
    (verbal_avg + motor_avg + eyes_avg)::int as gcs_avg,
    (verbal_max + motor_max + eyes_max)::int as gcs_max

from pivot2

);
