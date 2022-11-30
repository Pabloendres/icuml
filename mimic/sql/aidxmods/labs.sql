-- --------------------------------------------------------------------------------------------------------------------
-- Labs
-- --------------------------------------------------------------------------------------------------------------------
-- Parámetros de laboratorio: exámenes de sangre, orina, etc.

-- Parámetro                        UOM                     Rango         
-- -----------------------------------------------------------------------
-- Presión arterial de O2           mmHg                    ]0, -- [
-- Presión arterial de CO2          mmHg                    ]0, 200[
-- Bilirrubina                      mg/mL                   ]0, 100[
-- Creatinina                       mg/mL                   ]0, 100[
-- Nitrógeno uréico en sangre       mg/mL                   ]0, 200[
-- Hematocrito                      %                       [0, 100]
-- Bicarbonato en sangre            mEq/L                   ]0, 100[
-- pH                               --                      [6, 8[
-- Plaquetas                        10^3/mm2                ]0, -- [
-- Potasio                          mEq/L                   ]0, -- [
-- Sodio                            mEq/L                   ]0, -- [
-- Cloro                            mEq/L                   ]0, -- [
-- Magnesio                         mEq/L                   ]0, -- [
-- Conteo de leucocitos             10^3/mcL                ]0, -- [
-- Linfocitos                       %                       [0, 100]
-- Neutrófilos                      %                       [0, 100]
-- Aspartato aminotransferasa       U/L                     [0, -- [
-- Alanina aminotransferasa         U/L                     [0, -- [
-- Fosfatasa alcalina               U/L                     ]0, -- [
-- Albúmina                         g/dL                    ]0, -- [
-- Glucosa                          mg/dL                   ]0, -- [
-- Exceso de base                   mEq/L                   ]-500, 500[
-- Tiempo parcial de protrombina    s                       ]0, 200[
-- BNP                              pg/mL                   ]0, -- [
-- Fibrinógeno                      mg/dL                   ]0, -- [
-- Hemoglobina                      g/dL                    ]0, 30[
-- Lactato                          mg/dL                   ]0, 50[
-- -----------------------------------------------------------------------


set search_path to public, mimiciii;

drop table if exists aidxmods.labs;

create table aidxmods.labs as (
    
with pivot as (
    select 
        charttimes.subject_id as subject_id,
        charttimes.hadm_id as hadm_id, 
        charttimes.icustay_id as icustay_id,
        charttimes.los as los,
        case
            when itemid = 50821
                and valuenum::numeric >  0
                then valuenum::numeric
            else null end
        as pao2,
        case
            when itemid = 50818
                and valuenum::numeric >  0
                and valuenum::numeric <  100
                then valuenum::numeric
            else null end
        as paco2,
        case
            when itemid = 50885
                and valuenum::numeric >  0
                and valuenum::numeric <  100
                then valuenum::numeric
            else null end
        as bilirubin,
        case
            when itemid = 50912
                and valuenum::numeric >  0
                and valuenum::numeric <  100
                then valuenum::numeric
            else null end
        as creatinine,
        case
            when itemid = 51006
                and valuenum::numeric >  0
                and valuenum::numeric <  200
                then valuenum::numeric
            else null end
        as bun,
        case
            when itemid = 50810
                and valuenum::numeric >= 0
                and valuenum::numeric <= 100
                then valuenum::numeric
            else null end
        as hematocrit,
        case
            when itemid = 50803
                and valuenum::numeric >  0
                and valuenum::numeric <  100
                then valuenum::numeric
            else null end
        as bicarbonate,
        case
            when itemid = 50820
                and valuenum::numeric >= 6
                and valuenum::numeric <= 8
                then valuenum::numeric
            else null end
        as ph,
        case
            when itemid = 51265
                and valuenum::numeric >  0
                then valuenum::numeric
            else null end
        as platelets,
        case
            when itemid = 50971
                and valuenum::numeric >  0
                then valuenum::numeric
            else null end
        as potassium,
        case
            when itemid = 50983
                and valuenum::numeric >  0
                then valuenum::numeric
            else null end
        as sodium,
        case
            when itemid = 50806
                and valuenum::numeric >  0
                then valuenum::numeric
            else null end
        as chloride,
        case
            when itemid = 50960
                and valuenum::numeric >  0
                then valuenum::numeric
            else null end
        as magnesium,
        case
            when itemid in (51300, 51301)
                and valuenum::numeric >  0
                then valuenum::numeric
            else null end
        as wbc,
        case
            when itemid = 51244
                and valuenum::numeric >  0
                and valuenum::numeric <= 100
                then valuenum::numeric
            else null end
        as lymphocytes,
        case
            when itemid = 51256
                and valuenum::numeric >  0
                and valuenum::numeric <= 100
                then valuenum::numeric
            else null end
        as neutrophils,
        case
            when itemid = 50878
                and valuenum::numeric >  0
                then valuenum::numeric
            else null end
        as ast,
        case
            when itemid = 50861
                and valuenum::numeric >  0
                then valuenum::numeric
            else null end
        as alt,
        case
            when itemid = 50863
                and valuenum::numeric >  0
                then valuenum::numeric
            else null end
        as alp,
        case
            when itemid = 50862
                and valuenum::numeric >  0
                then valuenum::numeric
            else null end
        as albumin,
        case
            when itemid = 50809
                and valuenum::numeric >  0
                then valuenum::numeric
            else null end
        as glucose,
        case
            when itemid = 50802
                and valuenum::numeric >  -500
                and valuenum::numeric <  500
                then valuenum::numeric
            else null end
        as base_excess,
        case
            when itemid = 51275
                and valuenum::numeric >  0
                and valuenum::numeric <  200
                then valuenum::numeric
            else null end
        as ptt,
        case
            when itemid = 51214
                and valuenum::numeric >  0
                then valuenum::numeric
            else null end
        as fibrinogen,
        case
            when itemid = 51222
                and valuenum::numeric >  0
                and valuenum::numeric <  30
                then valuenum::numeric
            else null end
        as hemoglobin,
        case
            when itemid = 50813
                and valuenum::numeric >  0
                and valuenum::numeric <  50
                then valuenum::numeric
            else null end
         as lactate

    from firstday.charttimes charttimes
    
    left join labevents
        on labevents.hadm_id = charttimes.hadm_id
        and labevents.charttime >= charttimes.starttime
        and labevents.charttime <  charttimes.endtime
        and labevents.itemid in (
        --  itemid  |                label                |  category  | fluid
            50802, -- Base Excess                         | Blood Gas  | Blood
            50806, -- Chloride, Whole Blood               | Blood Gas  | Blood
            50809, -- Glucose                             | Blood Gas  | Blood
            50810, -- Hematocrit, Calculated              | Blood Gas  | Blood
            50811, -- Hemoglobin                          | Blood Gas  | Blood
            50813, -- Lactate                             | Blood Gas  | Blood
            50818, -- pCO2                                | Blood Gas  | Blood
            50821, -- pO2                                 | Blood Gas  | Blood
            50822, -- Potassium, Whole Blood              | Blood Gas  | Blood
            50824, -- Sodium, Whole Blood                 | Blood Gas  | Blood
            50820, -- pH                                  | Blood Gas  | Blood 
            50803, -- Calculated Bicarbonate, Whole Blood | Blood Gas  | Blood 
            50862, -- Albumin                             | Chemistry  | Blood
            50882, -- Bicarbonate                         | Chemistry  | Blood
            50885, -- Bilirubin, Total                    | Chemistry  | Blood
            50902, -- Chloride                            | Chemistry  | Blood
            50912, -- Creatinine                          | Chemistry  | Blood
            50931, -- Glucose                             | Chemistry  | Blood
            50971, -- Potassium                           | Chemistry  | Blood
            50983, -- Sodium                              | Chemistry  | Blood
            51006, -- Urea Nitrogen                       | Chemistry  | Blood
            50861, -- Alanine Aminotransferase (ALT)      | Chemistry  | Blood
            50878, -- Asparate Aminotransferase (AST)     | Chemistry  | Blood
            50960, -- Magnesium                           | Chemistry  | Blood
            50863, -- Alkaline Phosphatase                | Chemistry  | Blood
            51221, -- Hematocrit                          | Hematology | Blood
            51222, -- Hemoglobin                          | Hematology | Blood
            51265, -- Platelet Count                      | Hematology | Blood
            51275, -- PTT                                 | Hematology | Blood
            51300, -- WBC Count                           | Hematology | Blood
            51301, -- White Blood Cells                   | Hematology | Blood
            51214, -- Fibrinogen, Functional              | Hematology | Blood
            51244, -- Lymphocytes                         | Hematology | Blood
            51256, -- Neutrophils                         | Hematology | Blood 
            50828  -- Ventilation
        )
)

select
    subject_id,
    hadm_id, 
    icustay_id,
    los,
    round(min(pao2), 2) as pao2_min,
    round(avg(pao2), 2) as pao2_avg,
    round(max(pao2), 2) as pao2_max,
    round(min(paco2), 2) as paco2_min,
    round(avg(paco2), 2) as paco2_avg,
    round(max(paco2), 2) as paco2_max,
    round(min(bilirubin), 2) as bilirubin_min,
    round(avg(bilirubin), 2) as bilirubin_avg,
    round(max(bilirubin), 2) as bilirubin_max,
    round(min(creatinine), 2) as creatinine_min,
    round(avg(creatinine), 2) as creatinine_avg,
    round(max(creatinine), 2) as creatinine_max,
    round(min(bun), 2) as bun_min,
    round(avg(bun), 2) as bun_avg,
    round(max(bun), 2) as bun_max,
    round(min(hematocrit), 2) as hematocrit_min,
    round(avg(hematocrit), 2) as hematocrit_avg,
    round(max(hematocrit), 2) as hematocrit_max,
    round(min(bicarbonate), 2) as bicarbonate_min,
    round(avg(bicarbonate), 2) as bicarbonate_avg,
    round(max(bicarbonate), 2) as bicarbonate_max,
    round(min(ph), 2) as ph_min,
    round(avg(ph), 2) as ph_avg,
    round(max(ph), 2) as ph_max,
    round(min(platelets), 2) as platelets_min,
    round(avg(platelets), 2) as platelets_avg,
    round(max(platelets), 2) as platelets_max,
    round(min(potassium), 2) as potassium_min,
    round(avg(potassium), 2) as potassium_avg,
    round(max(potassium), 2) as potassium_max,
    round(min(sodium), 2) as sodium_min,
    round(avg(sodium), 2) as sodium_avg,
    round(max(sodium), 2) as sodium_max,
    round(min(chloride), 2) as chloride_min,
    round(avg(chloride), 2) as chloride_avg,
    round(max(chloride), 2) as chloride_max,
    round(min(magnesium), 2) as magnesium_min,
    round(avg(magnesium), 2) as magnesium_avg,
    round(max(magnesium), 2) as magnesium_max,
    round(min(wbc), 2) as wbc_min,
    round(avg(wbc), 2) as wbc_avg,
    round(max(wbc), 2) as wbc_max,
    round(min(lymphocytes), 2) as lymphocytes_min,
    round(avg(lymphocytes), 2) as lymphocytes_avg,
    round(max(lymphocytes), 2) as lymphocytes_max,
    round(min(neutrophils), 2) as neutrophils_min,
    round(avg(neutrophils), 2) as neutrophils_avg,
    round(max(neutrophils), 2) as neutrophils_max,
    round(min(ast), 2) as ast_min,
    round(avg(ast), 2) as ast_avg,
    round(max(ast), 2) as ast_max,
    round(min(alt), 2) as alt_min,
    round(avg(alt), 2) as alt_avg,
    round(max(alt), 2) as alt_max,
    round(min(alp), 2) as alp_min,
    round(avg(alp), 2) as alp_avg,
    round(max(alp), 2) as alp_max,
    round(min(albumin), 2) as albumin_min,
    round(avg(albumin), 2) as albumin_avg,
    round(max(albumin), 2) as albumin_max,
    round(min(glucose), 2) as glucose_min,
    round(avg(glucose), 2) as glucose_avg,
    round(max(glucose), 2) as glucose_max,
    round(min(base_excess), 2) as base_excess_min,
    round(avg(base_excess), 2) as base_excess_avg,
    round(max(base_excess), 2) as base_excess_max,
    round(min(ptt), 2) as ptt_min,
    round(avg(ptt), 2) as ptt_avg,
    round(max(ptt), 2) as ptt_max,
    round(min(fibrinogen), 2) as fibrinogen_min,
    round(avg(fibrinogen), 2) as fibrinogen_avg,
    round(max(fibrinogen), 2) as fibrinogen_max,
    round(min(hemoglobin), 2) as hemoglobin_min,
    round(avg(hemoglobin), 2) as hemoglobin_avg,
    round(max(hemoglobin), 2) as hemoglobin_max,
    round(min(lactate), 2) as lactate_min,
    round(avg(lactate), 2) as lactate_avg,
    round(max(lactate), 2) as lactate_max

from pivot

group by pivot.subject_id, pivot.hadm_id, pivot.icustay_id, pivot.los
order by pivot.subject_id, pivot.hadm_id, pivot.icustay_id, pivot.los

);
