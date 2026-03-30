select * from preguntas_y_aclaraciones limit 100;

select * from process_data limit 100;

SELECT 
    t1.contract_id,
    t2.preguntas_y_aclaraciones
FROM  process_data t1
JOIN preguntas_y_aclaraciones t2 ON t1.contract_id = t2.contract_id
WHERE EXISTS (
    SELECT 1
    FROM jsonb_array_elements(t1.data_json->'fechas') AS f(elemento)
    WHERE f.elemento->>'fecha_name' = 'fecha_de_publicacion'
      AND f.elemento->>'fecha_value' LIKE '2022-%'
);

SELECT 
    t1.contract_id,
    t2.preguntas_y_aclaraciones
FROM process_data t1
JOIN preguntas_y_aclaraciones t2 ON t1.contract_id = t2.contract_id
WHERE EXISTS (
    SELECT 1
    FROM jsonb_array_elements(t1.data_json->'fechas') AS f(elemento)
    WHERE f.elemento->>'fecha_name' = 'fecha_de_publicacion'
      AND f.elemento->>'fecha_value' LIKE '2022-%'
)
AND jsonb_typeof(t2.preguntas_y_aclaraciones) = 'array';


SELECT 
    t1.contract_id,
    (kv.value ->> 'pregunta_aclaracion') AS pregunta
FROM 
    process_data t1
JOIN 
    preguntas_y_aclaraciones t2 ON t1.contract_id = t2.contract_id,
    jsonb_array_elements(t2.preguntas_y_aclaraciones) AS array_item,
    -- Aquí introducimos la validación (CASE) para evitar el error 22023
    jsonb_each(
        CASE 
            WHEN jsonb_typeof(array_item) = 'object' THEN array_item 
            ELSE '{}'::jsonb 
        END
    ) AS kv
WHERE EXISTS (
    SELECT 1
    FROM jsonb_array_elements(t1.data_json->'fechas') AS f(elemento)
    WHERE f.elemento->>'fecha_name' = 'fecha_de_publicacion'
      AND f.elemento->>'fecha_value' LIKE '2022-%'
)
AND jsonb_typeof(t2.preguntas_y_aclaraciones) = 'array'
AND kv.key LIKE 'pregunta\_%' 
-- Por seguridad, asegurarnos también de que el valor interno sea un objeto antes de extraer campos
AND jsonb_typeof(kv.value) = 'object'
AND (kv.value ->> 'pregunta_aclaracion') IS NOT NULL
AND trim(kv.value ->> 'pregunta_aclaracion') <> '';


---SIE y GEN en general----

SELECT 
    t1.contract_id,
    (kv.value ->> 'pregunta_aclaracion') AS pregunta
FROM 
    process_data t1
JOIN 
    preguntas_y_aclaraciones t2 ON t1.contract_id = t2.contract_id,
    jsonb_array_elements(t2.preguntas_y_aclaraciones) AS array_item,
    -- Aquí introducimos la validación (CASE) para evitar el error 22023
    jsonb_each(
        CASE 
            WHEN jsonb_typeof(array_item) = 'object' THEN array_item 
            ELSE '{}'::jsonb 
        END
    ) AS kv
WHERE EXISTS (
    SELECT 1
    FROM jsonb_array_elements(t1.data_json->'fechas') AS f(elemento)
    WHERE f.elemento->>'fecha_name' = 'fecha_de_publicacion'
      AND f.elemento->>'fecha_value' LIKE '2022-%'
)
AND jsonb_typeof(t2.preguntas_y_aclaraciones) = 'array'
AND kv.key LIKE 'pregunta\_%' 
-- Por seguridad, asegurarnos también de que el valor interno sea un objeto antes de extraer campos
AND jsonb_typeof(kv.value) = 'object'
AND (kv.value ->> 'pregunta_aclaracion') IS NOT NULL
AND trim(kv.value ->> 'pregunta_aclaracion') <> ''
-- NUEVA LÍNEA: Excluir las que son solamente la palabra ACLARACION (con o sin tilde)
AND upper(trim(kv.value ->> 'pregunta_aclaracion')) NOT IN ('ACLARACION', 'ACLARACIÓN');


--SIE o GEN especificados--

WITH filtro AS (
    -- Cambia 'SIE' por 'GEN' si quieres ese tipo
    SELECT 'SIE'::text AS tipo_proceso
),
procesos AS (
    SELECT
        t1.contract_id,
        t1.data_json,
        t1.data_json -> 'descripcion' ->> 'codigo' AS codigo_proceso,
        upper(
            translate(
                t1.data_json -> 'descripcion' ->> 'codigo',
                '–—−‑‑',
                '-----'              -- normaliza guiones Unicode
            )
        ) AS codigo_norm
    FROM process_data t1
)
SELECT
    pr.contract_id,
    pr.codigo_proceso,
    kv.value ->> 'pregunta_aclaracion' AS pregunta
FROM procesos pr
JOIN preguntas_y_aclaraciones t2
  ON pr.contract_id = t2.contract_id,
     jsonb_array_elements(t2.preguntas_y_aclaraciones) AS array_item,
     jsonb_each(
         CASE
             WHEN jsonb_typeof(array_item) = 'object' THEN array_item
             ELSE '{}'::jsonb
         END
     ) AS kv
JOIN filtro f ON TRUE
WHERE EXISTS (
        SELECT 1
        FROM jsonb_array_elements(pr.data_json -> 'fechas') AS f2(elemento)
        WHERE f2.elemento ->> 'fecha_name'  = 'fecha_de_publicacion'
          AND f2.elemento ->> 'fecha_value' LIKE '2022-%'
      )
  AND jsonb_typeof(t2.preguntas_y_aclaraciones) = 'array'
  AND kv.key LIKE 'pregunta\_%'
  AND jsonb_typeof(kv.value) = 'object'
  AND kv.value ->> 'pregunta_aclaracion' IS NOT NULL
  AND trim(kv.value ->> 'pregunta_aclaracion') <> ''
  AND upper(trim(kv.value ->> 'pregunta_aclaracion')) NOT IN ('ACLARACION', 'ACLARACIÓN')
  AND (
        -- GEN: mismo comportamiento original
        (f.tipo_proceso ILIKE 'GEN' AND pr.codigo_norm LIKE 'GEN-%')
        OR
        -- SIE: incluye SIE- / SIE_ pero excluye “reverse” (prefijos que terminan en R antes de SIE)
        (
            f.tipo_proceso ILIKE 'SIE'
            AND pr.codigo_norm !~ '^R[[:alnum:]._-]*SIE'
            AND (
                   pr.codigo_norm LIKE '%SIE-%'
                OR pr.codigo_norm LIKE '%SIE_%'
                OR pr.codigo_norm LIKE '%SIE.%'
                OR pr.codigo_norm LIKE '%SIE/%'
            )
        )
      );



-- Ni SIE ni GEN ---
WITH procesos AS (
    SELECT
        t1.contract_id,
        t1.data_json,
        t1.data_json -> 'descripcion' ->> 'codigo' AS codigo_proceso,
        upper(
            translate(
                t1.data_json -> 'descripcion' ->> 'codigo',
                '–—−‑‑',
                '-----'
            )
        ) AS codigo_norm
    FROM process_data t1
)
SELECT
    pr.contract_id,
    pr.codigo_proceso,
    kv.value ->> 'pregunta_aclaracion' AS pregunta
FROM procesos pr
JOIN preguntas_y_aclaraciones t2
      ON pr.contract_id = t2.contract_id,
     jsonb_array_elements(t2.preguntas_y_aclaraciones) AS array_item,
     jsonb_each(
         CASE
             WHEN jsonb_typeof(array_item) = 'object' THEN array_item
             ELSE '{}'::jsonb
         END
     ) AS kv
WHERE EXISTS (
        SELECT 1
        FROM jsonb_array_elements(pr.data_json -> 'fechas') AS f2(elemento)
        WHERE f2.elemento ->> 'fecha_name'  = 'fecha_de_publicacion'
          AND f2.elemento ->> 'fecha_value' LIKE '2022-%'
      )
  AND jsonb_typeof(t2.preguntas_y_aclaraciones) = 'array'
  AND kv.key LIKE 'pregunta\_%'
  AND jsonb_typeof(kv.value) = 'object'
  AND kv.value ->> 'pregunta_aclaracion' IS NOT NULL
  AND trim(kv.value ->> 'pregunta_aclaracion') <> ''
  AND upper(trim(kv.value ->> 'pregunta_aclaracion')) NOT IN ('ACLARACION', 'ACLARACIÓN')
  AND NOT (
        -- GEN con prefijo estándar
        pr.codigo_norm LIKE 'GEN-%'
        OR
        -- SIE con separadores válidos (sin “reverse”)
        (
            pr.codigo_norm !~ '^R[[:alnum:]._-]*SIE'
            AND (
                   pr.codigo_norm LIKE '%SIE-%'
                OR pr.codigo_norm LIKE '%SIE_%'
                OR pr.codigo_norm LIKE '%SIE.%'
                OR pr.codigo_norm LIKE '%SIE/%'
            )
        )
      );
