-- --------------------------------------------------------------------------------------------------------------------
-- Labs
-- --------------------------------------------------------------------------------------------------------------------
-- Parámetros de laboratorio: exámenes de sangre, orina, etc.

-- Parámetro                        UOM                     Rango         
-- -----------------------------------------------------------------------
-- Fracción inspirada de oxígeno    %                       [21, 100]
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
-- Fibrinógeno                      mg/dL                   ]0, 2500[
-- Hemoglobina                      g/dL                    ]0, 30[
-- Lactato                          mg/dL                   ]0, 50[
-- -----------------------------------------------------------------------


set search_path to public, eicu_crd;

drop table if exists aidxmods.labs;

create table aidxmods.labs as (
    
with pivot as (
    select 
        charttimes.uniquepid as uniquepid,
        charttimes.patienthealthsystemstayid as patienthealthsystemstayid,
        charttimes.patientunitstayid as patientunitstayid,
        charttimes.los as los,
        case
            when lab.labname = 'FiO2'
                and lab.labresult >  0.21
                and lab.labresult <= 1.00
                then lab.labresult * 100
            when lab.labname = 'FiO2'
                and lab.labresult >  21
                and lab.labresult <= 100
                then lab.labresult
            else null end
        as fio2,
        case
            when lab.labname = 'paO2'
                and lab.labresult >  0
                then lab.labresult
            else null end
        as pao2, 
        case
            when lab.labname = 'paCO2' 
                and lab.labresult >  0
                and lab.labresult <  100
                then lab.labresult
            else null end
        as paco2, 
        case
            when lab.labname = 'total bilirubin' 
                and lab.labresult >  0
                and lab.labresult <  100
                then lab.labresult
            else null end
        as bilirubin, 
        case
            when lab.labname = 'creatinine' 
                and lab.labresult >  0
                and lab.labresult <  100
                then lab.labresult
            else null end
        as creatinine, 
        case
            when lab.labname = 'BUN' 
                and lab.labresult >  0
                and lab.labresult <  200
            then lab.labresult
            else null end
        as bun, 
        case
            when lab.labname = 'Hct' 
                and lab.labresult >= 0 
                and lab.labresult <= 100
                then lab.labresult
            else null end
        as hematocrit, 
        case
            when lab.labname in ('bicarbonate', 'HCO3')
                and lab.labresult >  0
                and lab.labresult <  100
                then lab.labresult
            else null end
        as bicarbonate, 
        case
            when lab.labname = 'pH'
                and lab.labresult >= 6
                and lab.labresult <= 8
                then lab.labresult
            else null end
        as ph, 
        case
            when lab.labname = 'platelets x 1000' 
                and lab.labresult >  0 
                then lab.labresult
            else null end
        as platelets, 
        case
            when lab.labname = 'potassium' 
                and lab.labresult > 0
                then lab.labresult
            else null end
        as potassium, 
        case
            when lab.labname = 'sodium' 
                and lab.labresult >  0
                then lab.labresult
            else null end
        as sodium, 
        case
            when lab.labname = 'chloride' 
                and lab.labresult >  0
                then lab.labresult
            else null end
        as chloride,
        case
            when lab.labname = 'magnesium' 
                and lab.labresult >  0
                then lab.labresult
            else null end
        as magnesium,
        case
            when lab.labname = 'WBC x 1000' 
                and lab.labresult >  0
                then lab.labresult
            else null end
        as wbc,
        case
            when lab.labname = '-lymphs' 
                and lab.labresult >  0 
                and lab.labresult <= 100
                then lab.labresult
            else null end
        as lymphocytes,
        case
            when lab.labname = '-polys' 
                and lab.labresult >  0 
                and lab.labresult <= 100
                then lab.labresult
            else null end
        as neutrophils, 
        case
            when lab.labname = 'AST (SGOT)' 
                and lab.labresult >  0
                then lab.labresult
            else null end
        as ast,
        case
            when lab.labname = 'ALT (SGPT)' 
                and lab.labresult >  0
                then lab.labresult
            else null end
        as alt,
        case
            when lab.labname = 'alkaline phos.' 
                and lab.labresult >  0
                then lab.labresult
            else null end
        as alp,
        case
            when lab.labname = 'albumin' 
                and lab.labresult >  0
                then lab.labresult
            else null end
        as albumin, 
        case
            when lab.labname in ('bedside glucose', 'glucose') 
                and lab.labresult >  0
                then lab.labresult
            else null end
        as glucose, 
        case
            when lab.labname = 'Base Excess' 
                and lab.labresult >  -500
                and lab.labresult <  500
                then lab.labresult
            else null end
        as base_excess, 
        case
            when lab.labname = 'PTT'
                and lab.labresult >  0
                and lab.labresult <  200
            then lab.labresult
            else null end
        as ptt,
        case
            when lab.labname = 'BNP' 
                and lab.labresult >  0
                then lab.labresult
            else null end
        as bnp,
        case
            when lab.labname = 'fibrinogen' 
                and lab.labresult >  0
                then lab.labresult
            else null end
        as fibrinogen,
        case
            when lab.labname = 'Hgb' 
                and lab.labresult >  0
                and lab.labresult <  30
                then lab.labresult
            else null end
        as hemoglobin, 
        case
            when lab.labname = 'lactate' 
                and lab.labresult >  0 
                and lab.labresult <  50
                then lab.labresult
            else null end
        as lactate

    from firstday.charttimes charttimes

    left join lab
        on lab.patientunitstayid = charttimes.patientunitstayid
        and lab.labresultoffset >= charttimes.starttime 
        and lab.labresultoffset <  charttimes.endtime

)

select
    uniquepid,
    patienthealthsystemstayid,
    patientunitstayid,
    los,
    round(min(fio2), 2) as fio2_min,
    round(avg(fio2), 2) as fio2_avg,
    round(max(fio2), 2) as fio2_max,
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
    round(min(bnp), 2) as bnp_min,
    round(avg(bnp), 2) as bnp_avg,
    round(max(bnp), 2) as bnp_max,
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

group by pivot.uniquepid, pivot.patienthealthsystemstayid, pivot.patientunitstayid, pivot.los

);
