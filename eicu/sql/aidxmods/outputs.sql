-- --------------------------------------------------------------------------------------------------------------------
-- Outputs
-- --------------------------------------------------------------------------------------------------------------------
-- Signos vitales.

-- Parámetro                        UOM                     Rango         
-- -----------------------------------------------------------------------
-- Salida de orina                  mL                      [0, -- [
-- -----------------------------------------------------------------------


set search_path to public, eicu_crd;

drop table if exists aidxmods.outputs;

create table aidxmods.outputs as (

select
    charttimes.uniquepid as uniquepid,
    charttimes.patienthealthsystemstayid as patienthealthsystemstayid,
    charttimes.patientunitstayid as patientunitstayid,
    charttimes.los as los,
    round(sum(case
        when cellpath like 'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|Urine Output%'
            or cellpath like 'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|foley%'
            or cellpath like 'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|Output%Urinary Catheter%'
            or cellpath like 'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|Output%Urethral Catheter%'
            or cellpath like 'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|Output%External Urethral%'
            or cellpath like 'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|Urinary Catheter Output%'
            or cellpath in (
                'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|Urine',
                'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|3 way foley',
                'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|3 Way Foley',
                'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|Actual Urine',
                'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|Adjusted total UO NOC end shift',
                'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|BRP (urine)',
                'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|BRP (Urine)',
                'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|condome cath urine',
                'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|diaper urine',
                'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|inc of urine',
                'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|incontient urine',
                'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|Incontient urine',
                'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|Incontient Urine',
                'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|incontinence of urine',
                'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|Incontinence-urine',
                'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|incontinence/ voids urine',
                'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|incontinent of urine',
                'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|INCONTINENT OF URINE',
                'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|Incontinent UOP',
                'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|incontinent urine',
                'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|Incontinent (urine)',
                'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|Incontinent Urine',
                'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|incontinent urine counts',
                'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|incont of urine',
                'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|incont. of urine',
                'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|incont. of urine count',
                'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|incont. of urine count',
                'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|incont urine',
                'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|incont. urine',
                'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|Incont. urine',
                'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|Incont. Urine',
                'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|inc urine',
                'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|inc. urine',
                'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|Inc. urine',
                'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|Inc Urine',
                'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|indwelling foley',
                'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|Indwelling Foley',
                'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|Straight Catheter-Foley',
                'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|Straight Catheterization Urine',
                'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|Straight Cath UOP',
                'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|straight cath urine',
                'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|Straight Cath Urine',
                'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|strait cath Urine',
                'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|Suprapubic Urine Output',
                'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|true urine',
                'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|True Urine',
                'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|True Urine out',
                'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|unmeasured urine',
                'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|Unmeasured Urine',
                'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|unmeasured urine output',
                'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|Urethal Catheter',
                'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|Urethral Catheter',
                'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|urinary output 7AM - 7 PM',
                'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|urinary output 7AM-7PM',
                'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|urine',
                'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|URINE',
                'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|URINE',
                'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|URINE CATHETER',
                'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|Intermittent/Straight Cath (mL)',
                'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|straightcath',
                'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|straight cath',
                'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|Straight cath',
                'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|Straight  cath',
                'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|Straight Cath',
                'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|Straight  Cath',
                'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|Straight Cath',
                'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|Straight Cath''d',
                'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|straight cath daily',
                'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|straight cathed',
                'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|Straight Cathed',
                'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|Straight Catheter-Foley',
                'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|Straight catheterization',
                'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|Straight Catheterization Urine',
                'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|Straight Catheter Output',
                'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|Straight Catheter-Straight Catheter',
                'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|straight cath ml''s',
                'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|Straight cath ml''s',
                'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|Straight Cath Q6hrs',
                'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|Straight caths',
                'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|Straight Cath UOP',
                'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|straight cath urine',
                'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|Straight Cath Urine',
                'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|Urine-straight cath',
                'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|Urine Straight Cath',
                'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|Condom Catheter',
                'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|condom catheter',
                'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|condome cath urine',
                'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|condom cath',
                'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|Condom Cath',
                'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|CONDOM CATHETER OUTPUT',
                'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|Urine via condom catheter',
                'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|Urine-foley',
                'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|Urine- foley',
                'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|Urine- Foley',
                'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|Urine foley catheter',
                'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|Urine, L neph:',
                'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|Urine (measured)',
                'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|Urine, R neph:',
                'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|Urine-straight cath',
                'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|Urine Straight Cath',
                'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|urine (void)',
                'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|Urine- void',
                'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|Urine, void:')
            and cellvaluenumeric > 0
            then cellvaluenumeric
        else 0 end
    ), 2) as urineoutput
    
from firstday.charttimes charttimes

left join intakeoutput
    on intakeoutput.patientunitstayid = charttimes.patientunitstayid
    and intakeoutput.intakeoutputoffset >= charttimes.starttime
    and intakeoutput.intakeoutputoffset <  charttimes.endtime
    and intakeoutput.cellpath like 'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|%'

group by charttimes.uniquepid, charttimes.patienthealthsystemstayid, charttimes.patientunitstayid, charttimes.los
    
);
