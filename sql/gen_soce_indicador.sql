SELECT DISTINCT 
    b.contract_id,
    b.codigo,
    sf.fecha_value::date AS fecha_publicacion,
    CASE SUBSTRING(b.estado_del_proceso::text, 1, 1)
        WHEN '{' THEN b.estado_del_proceso::text
        ELSE ('{"estado_del_proceso":"' || b.estado_del_proceso::text) || '"}'
    END::jsonb ->> 'estado_del_proceso' AS estado_del_proceso,
    b.tipo_de_contratacion
FROM soce_descripcion b
JOIN soce_fechas sf ON b.contract_id::text = sf.contract_id::text
WHERE sf.fecha_name = 'fecha_de_publicacion'
AND (
    -- Vía 1: Publicación Especial con código GEN válido
    (b.tipo_de_contratacion = 'Publicación Especial' AND valida_codigo_gen_pe(b.codigo))
    OR
    -- Vía 2: Empresas Públicas por definición
    (b.tipo_de_contratacion = 'Empresas Públicas, Mercantiles o Subsidiarias')
)
ORDER BY fecha_publicacion;

--con filtro de fecha---
SELECT DISTINCT 
    b.contract_id,
    b.codigo,
    sf.fecha_value::date AS fecha_publicacion,
    CASE SUBSTRING(b.estado_del_proceso::text, 1, 1)
        WHEN '{' THEN b.estado_del_proceso::text
        ELSE ('{"estado_del_proceso":"' || b.estado_del_proceso::text) || '"}'
    END::jsonb ->> 'estado_del_proceso' AS estado_del_proceso,
    b.tipo_de_contratacion
FROM soce_descripcion b
JOIN soce_fechas sf ON b.contract_id::text = sf.contract_id::text
WHERE sf.fecha_name = 'fecha_de_publicacion'
AND date_part('year', sf.fecha_value::date) = 2022
AND (
    (b.tipo_de_contratacion = 'Publicación Especial' AND valida_codigo_gen_pe(b.codigo))
    OR
    (b.tipo_de_contratacion = 'Empresas Públicas, Mercantiles o Subsidiarias')
)
ORDER BY fecha_publicacion;

-------extracción de datos--------

SELECT 
    z.contract_id,
    z.codigo,
    z.fecha_publicacion,
    z.estado_del_proceso,
    z.valor_preguntas,
    z.flag
FROM (
    SELECT 
        x.contract_id,
        x.codigo,
        x.estado_del_proceso,
        x.fecha_publicacion,
        ( SELECT jsonb_each.value -> 'pregunta_aclaracion'::text
               FROM jsonb_each(x.preguntas) jsonb_each(key, value)) AS valor_preguntas,
        to_tsvector('spanish'::regconfig, (( SELECT jsonb_each.value -> 'pregunta_aclaracion'::text
               FROM jsonb_each(x.preguntas) jsonb_each(key, value)))::text) @@ 
        to_tsquery('spanish'::regconfig, 
            'corrupción|direccionado|limitante|vulneración|ocultamiento|violación|incompleto|trato<->justo'
        ) AS flag
    FROM (
        SELECT DISTINCT 
            b.contract_id,
            b.codigo,
            sf.fecha_value::date AS fecha_publicacion,
            CASE SUBSTRING(b.estado_del_proceso::text, 1, 1)
                WHEN '{' THEN b.estado_del_proceso::text
                ELSE ('{"estado_del_proceso":"' || b.estado_del_proceso::text) || '"}'
            END::jsonb ->> 'estado_del_proceso' AS estado_del_proceso,
            jsonb_array_elements(a.preguntas_y_aclaraciones) AS preguntas,
            is_valid_json(jsonb_array_elements_text(a.preguntas_y_aclaraciones)) AS condicion
        FROM soce_descripcion b
        JOIN soce_fechas sf ON b.contract_id::text = sf.contract_id::text
        JOIN preguntas_y_aclaraciones a ON b.contract_id::integer = a.contract_id
        WHERE jsonb_typeof(a.preguntas_y_aclaraciones) = 'array'
        AND sf.fecha_name = 'fecha_de_publicacion'
        AND date_part('year', sf.fecha_value::date) = 2022  -- ← filtro año
        AND b.tipo_de_contratacion = 'Publicación Especial'
        AND valida_codigo_gen_pe(b.codigo)

        UNION

        SELECT DISTINCT 
            b.contract_id,
            b.codigo,
            sf.fecha_value::date AS fecha_publicacion,
            CASE SUBSTRING(b.estado_del_proceso::text, 1, 1)
                WHEN '{' THEN b.estado_del_proceso::text
                ELSE ('{"estado_del_proceso":"' || b.estado_del_proceso::text) || '"}'
            END::jsonb ->> 'estado_del_proceso' AS estado_del_proceso,
            jsonb_array_elements(a.preguntas_y_aclaraciones) AS preguntas,
            is_valid_json(jsonb_array_elements_text(a.preguntas_y_aclaraciones)) AS condicion
        FROM soce_descripcion b
        JOIN soce_fechas sf ON b.contract_id::text = sf.contract_id::text
        JOIN preguntas_y_aclaraciones a ON b.contract_id::integer = a.contract_id
        WHERE jsonb_typeof(a.preguntas_y_aclaraciones) = 'array'
        AND sf.fecha_name = 'fecha_de_publicacion'
        AND date_part('year', sf.fecha_value::date) = 2022 
        AND b.tipo_de_contratacion = 'Empresas Públicas, Mercantiles o Subsidiarias'
    ) x
    WHERE x.condicion IS TRUE 
    AND x.estado_del_proceso = ANY (ARRAY[
        'Adjudicada', 'Cancelado', 'Desierta', 'Finalizada', 'Por Adjudicar',
        'Preguntas, Respuestas y Aclaraciones', 'Audiencia de Preguntas y Aclaraciones',
        'Terminado Unilateralmente', 'En Curso', 'Suspendido'
    ])
) z
WHERE z.valor_preguntas IS NOT NULL
ORDER BY z.fecha_publicacion;

--extraccion 2--


 SELECT 
    z.contract_id,
    z.codigo,
    z.fecha_publicacion,
    z.estado_del_proceso,
    z.valor_preguntas,
    z.flag AS es_acusatoria,
    CASE 
        WHEN z.flag THEN 'Acusatoria'
        ELSE 'No Acusatoria'
    END AS clasificacion
FROM (
    SELECT 
        x.contract_id,
        x.codigo,
        x.estado_del_proceso,
        x.fecha_publicacion,
        (SELECT jsonb_each.value -> 'pregunta_aclaracion'::text
           FROM jsonb_each(x.preguntas) jsonb_each(key, value)) AS valor_preguntas,
        to_tsvector('spanish'::regconfig, ((SELECT jsonb_each.value -> 'pregunta_aclaracion'::text
           FROM jsonb_each(x.preguntas) jsonb_each(key, value)))::text) 
            @@ to_tsquery('spanish'::regconfig, 'corrupción|direccionado|limitante|vulneración|ocultamiento|violación|incompleto|trato<->justo'::text) AS flag
    FROM (
        SELECT DISTINCT 
            b.contract_id,
            b.codigo,
            sf.fecha_value::date AS fecha_publicacion,
            CASE substring(b.estado_del_proceso::text, 1, 1)
                WHEN '{'::text THEN b.estado_del_proceso::text
                ELSE ('{"estado_del_proceso":"'::text || b.estado_del_proceso::text) || '"}'::text
            END::jsonb ->> 'estado_del_proceso'::text AS estado_del_proceso,
            jsonb_array_elements(a.preguntas_y_aclaraciones) AS preguntas,
            is_valid_json(jsonb_array_elements_text(a.preguntas_y_aclaraciones)) AS condicion
        FROM soce_descripcion b
            JOIN soce_fechas sf ON b.contract_id::text = sf.contract_id::text
            JOIN preguntas_y_aclaraciones a ON b.contract_id::integer = a.contract_id
        WHERE jsonb_typeof(a.preguntas_y_aclaraciones) = 'array'::text 
            AND sf.fecha_name::text = 'fecha_de_publicacion'::text 
            AND b.tipo_de_contratacion::text = 'Publicación Especial'::text 
            AND valida_codigo_gen_pe(b.codigo)
        UNION
        SELECT DISTINCT 
            b.contract_id,
            b.codigo,
            sf.fecha_value::date AS fecha_publicacion,
            CASE substring(b.estado_del_proceso::text, 1, 1)
                WHEN '{'::text THEN b.estado_del_proceso::text
                ELSE ('{"estado_del_proceso":"'::text || b.estado_del_proceso::text) || '"}'::text
            END::jsonb ->> 'estado_del_proceso'::text AS estado_del_proceso,
            jsonb_array_elements(a.preguntas_y_aclaraciones) AS preguntas,
            is_valid_json(jsonb_array_elements_text(a.preguntas_y_aclaraciones)) AS condicion
        FROM soce_descripcion b
            JOIN soce_fechas sf ON b.contract_id::text = sf.contract_id::text
            JOIN preguntas_y_aclaraciones a ON b.contract_id::integer = a.contract_id
        WHERE jsonb_typeof(a.preguntas_y_aclaraciones) = 'array'::text 
            AND sf.fecha_name::text = 'fecha_de_publicacion'::text 
            AND b.tipo_de_contratacion::text = 'Empresas Públicas, Mercantiles o Subsidiarias'::text
    ) x
    WHERE x.condicion IS TRUE 
        AND (x.estado_del_proceso = ANY (ARRAY[
            'Adjudicada'::text, 'Cancelado'::text, 'Desierta'::text, 
            'Finalizada'::text, 'Por Adjudicar'::text, 
            'Preguntas, Respuestas y Aclaraciones'::text, 
            'Audiencia de Preguntas y Aclaraciones'::text, 
            'Terminado Unilateralmente'::text, 'En Curso'::text, 'Suspendido'::text
        ]))
) z
ORDER BY z.fecha_publicacion, z.contract_id;

--filtrado por 2022 GEN--

SELECT 
    z.contract_id,
    z.codigo,
    z.fecha_publicacion,
    z.estado_del_proceso,
    z.valor_preguntas,
    z.flag AS es_acusatoria,
    CASE 
        WHEN z.flag THEN 'Acusatoria'
        ELSE 'No Acusatoria'
    END AS clasificacion
FROM (
    SELECT 
        x.contract_id,
        x.codigo,
        x.estado_del_proceso,
        x.fecha_publicacion,
        (SELECT jsonb_each.value -> 'pregunta_aclaracion'::text
           FROM jsonb_each(x.preguntas) jsonb_each(key, value)) AS valor_preguntas,
        to_tsvector('spanish'::regconfig, ((SELECT jsonb_each.value -> 'pregunta_aclaracion'::text
           FROM jsonb_each(x.preguntas) jsonb_each(key, value)))::text) 
            @@ to_tsquery('spanish'::regconfig, 'corrupción|direccionado|limitante|vulneración|ocultamiento|violación|incompleto|trato<->justo'::text) AS flag
    FROM (
        SELECT DISTINCT 
            b.contract_id,
            b.codigo,
            sf.fecha_value::date AS fecha_publicacion,
            CASE substring(b.estado_del_proceso::text, 1, 1)
                WHEN '{'::text THEN b.estado_del_proceso::text
                ELSE ('{"estado_del_proceso":"'::text || b.estado_del_proceso::text) || '"}'::text
            END::jsonb ->> 'estado_del_proceso'::text AS estado_del_proceso,
            jsonb_array_elements(a.preguntas_y_aclaraciones) AS preguntas,
            is_valid_json(jsonb_array_elements_text(a.preguntas_y_aclaraciones)) AS condicion
        FROM soce_descripcion b
            JOIN soce_fechas sf ON b.contract_id::text = sf.contract_id::text
            JOIN preguntas_y_aclaraciones a ON b.contract_id::integer = a.contract_id
        WHERE jsonb_typeof(a.preguntas_y_aclaraciones) = 'array'::text 
            AND sf.fecha_name::text = 'fecha_de_publicacion'::text 
            AND b.tipo_de_contratacion::text = 'Publicación Especial'::text 
            AND valida_codigo_gen_pe(b.codigo)
        UNION
        SELECT DISTINCT 
            b.contract_id,
            b.codigo,
            sf.fecha_value::date AS fecha_publicacion,
            CASE "substring"(b.estado_del_proceso::text, 1, 1)
                WHEN '{'::text THEN b.estado_del_proceso::text
                ELSE ('{"estado_del_proceso":"'::text || b.estado_del_proceso::text) || '"}'::text
            END::jsonb ->> 'estado_del_proceso'::text AS estado_del_proceso,
            jsonb_array_elements(a.preguntas_y_aclaraciones) AS preguntas,
            is_valid_json(jsonb_array_elements_text(a.preguntas_y_aclaraciones)) AS condicion
        FROM soce_descripcion b
            JOIN soce_fechas sf ON b.contract_id::text = sf.contract_id::text
            JOIN preguntas_y_aclaraciones a ON b.contract_id::integer = a.contract_id
        WHERE jsonb_typeof(a.preguntas_y_aclaraciones) = 'array'::text 
            AND sf.fecha_name::text = 'fecha_de_publicacion'::text 
            AND b.tipo_de_contratacion::text = 'Empresas Públicas, Mercantiles o Subsidiarias'::text
    ) x
    WHERE x.condicion IS TRUE 
        AND (x.estado_del_proceso = ANY (ARRAY[
            'Adjudicada'::text, 'Cancelado'::text, 'Desierta'::text, 
            'Finalizada'::text, 'Por Adjudicar'::text, 
            'Preguntas, Respuestas y Aclaraciones'::text, 
            'Audiencia de Preguntas y Aclaraciones'::text, 
            'Terminado Unilateralmente'::text, 'En Curso'::text, 'Suspendido'::text
        ]))
) z
WHERE EXTRACT(YEAR FROM z.fecha_publicacion) = 2022 
ORDER BY z.fecha_publicacion, z.contract_id;

--SIE data--

SELECT 
    z.contract_id,
    z.codigo,
    z.fecha_publicacion,
    z.estado_del_proceso,
    p.presupuesto_referencial_total_sin_iva,
    p.monto_contrato,
    p.monto_adjudicacion,
    z.valor_preguntas,
    z.flag AS es_acusatoria,
    CASE 
        WHEN z.flag THEN 'Acusatoria'
        ELSE 'No Acusatoria'
    END AS clasificacion
FROM (
    SELECT 
        x.contract_id,
        x.codigo,
        x.estado_del_proceso,
        x.fecha_publicacion,
        (SELECT jsonb_each.value -> 'pregunta_aclaracion'::text
           FROM jsonb_each(x.preguntas) jsonb_each(key, value)) AS valor_preguntas,
        to_tsvector('spanish'::regconfig, ((SELECT jsonb_each.value -> 'pregunta_aclaracion'::text
           FROM jsonb_each(x.preguntas) jsonb_each(key, value)))::text) 
            @@ to_tsquery('spanish'::regconfig, 'corrupción|direccionado|limitante|vulneración|ocultamiento|violación|incompleto|trato<->justo'::text) AS flag
    FROM (
        SELECT 
            b.contract_id,
            b.codigo,
            sf.fecha_value::date AS fecha_publicacion,
            CASE substring(b.estado_del_proceso::text, 1, 1)
                WHEN '{'::text THEN b.estado_del_proceso::text
                ELSE ('{"estado_del_proceso":"'::text || b.estado_del_proceso::text) || '"}'::text
            END::jsonb ->> 'estado_del_proceso'::text AS estado_del_proceso,
            jsonb_array_elements(a.preguntas_y_aclaraciones) AS preguntas,
            is_valid_json(jsonb_array_elements_text(a.preguntas_y_aclaraciones)) AS condicion
        FROM soce_descripcion b
            JOIN soce_fechas sf ON b.contract_id::text = sf.contract_id::text
            JOIN preguntas_y_aclaraciones a ON b.contract_id::integer = a.contract_id
        WHERE jsonb_typeof(a.preguntas_y_aclaraciones) = 'array'::text 
            AND b.tipo_de_contratacion::text = 'Subasta Inversa Electrónica'::text 
            AND sf.fecha_name::text = 'fecha_de_publicacion'::text
    ) x
    WHERE x.condicion IS TRUE 
        AND (x.estado_del_proceso = ANY (ARRAY[
            'Preguntas, Respuestas y Aclaraciones'::text, 'Entrega de Propuesta'::text,
            'Convalidacion de Errores'::text, 'Calificación de Participantes'::text,
            'Oferta Inicial'::text, 'Puja'::text, 'Negociación'::text,
            'Reprogramación Puja'::text, 'Suspendido'::text, 'Por Adjudicar'::text,
            'Adjudicado - Registro de Contratos'::text, 'Adjudicada'::text,
            'Ejecución de Contrato'::text, 'En Recepción'::text, 'Finalizada'::text,
            'Terminado Unilateralmente'::text, 'Finalizado por mutuo acuerdo'::text,
            'Finalizado por disolución de la persona Jurídica'::text,
            'Finalizado a solicitud del contratista'::text, 'Cancelado'::text, 'Desierta'::text
        ]))
) z
LEFT JOIN mv_sie_procesos_datos p ON z.contract_id::text = p.contract_id::text 
    AND z.codigo::text = p.codigo::text
WHERE EXTRACT(YEAR FROM z.fecha_publicacion) = 2022
ORDER BY z.fecha_publicacion, z.contract_id;

--gen filtrado por aclaracion--

SELECT 
    z.contract_id,
    z.codigo,
    z.fecha_publicacion,
    z.estado_del_proceso,
    z.valor_preguntas,
    z.flag AS es_acusatoria,
    CASE 
        WHEN z.flag THEN 'Acusatoria'
        ELSE 'No Acusatoria'
    END AS clasificacion
FROM (
    SELECT 
        x.contract_id,
        x.codigo,
        x.estado_del_proceso,
        x.fecha_publicacion,
        (SELECT jsonb_each.value -> 'pregunta_aclaracion'::text
           FROM jsonb_each(x.preguntas) jsonb_each(key, value)) AS valor_preguntas,
        to_tsvector('spanish'::regconfig, ((SELECT jsonb_each.value -> 'pregunta_aclaracion'::text
           FROM jsonb_each(x.preguntas) jsonb_each(key, value)))::text) 
            @@ to_tsquery('spanish'::regconfig, 'corrupción|direccionado|limitante|vulneración|ocultamiento|violación|incompleto|trato<->justo'::text) AS flag
    FROM (
        SELECT DISTINCT 
            b.contract_id,
            b.codigo,
            sf.fecha_value::date AS fecha_publicacion,
            CASE substring(b.estado_del_proceso::text, 1, 1)
                WHEN '{'::text THEN b.estado_del_proceso::text
                ELSE ('{"estado_del_proceso":"'::text || b.estado_del_proceso::text) || '"}'::text
            END::jsonb ->> 'estado_del_proceso'::text AS estado_del_proceso,
            jsonb_array_elements(a.preguntas_y_aclaraciones) AS preguntas,
            is_valid_json(jsonb_array_elements_text(a.preguntas_y_aclaraciones)) AS condicion
        FROM soce_descripcion b
            JOIN soce_fechas sf ON b.contract_id::text = sf.contract_id::text
            JOIN preguntas_y_aclaraciones a ON b.contract_id::integer = a.contract_id
        WHERE jsonb_typeof(a.preguntas_y_aclaraciones) = 'array'::text 
            AND sf.fecha_name::text = 'fecha_de_publicacion'::text 
            AND b.tipo_de_contratacion::text = 'Publicación Especial'::text 
            AND valida_codigo_gen_pe(b.codigo)
        UNION
        SELECT DISTINCT 
            b.contract_id,
            b.codigo,
            sf.fecha_value::date AS fecha_publicacion,
            CASE "substring"(b.estado_del_proceso::text, 1, 1)
                WHEN '{'::text THEN b.estado_del_proceso::text
                ELSE ('{"estado_del_proceso":"'::text || b.estado_del_proceso::text) || '"}'::text
            END::jsonb ->> 'estado_del_proceso'::text AS estado_del_proceso,
            jsonb_array_elements(a.preguntas_y_aclaraciones) AS preguntas,
            is_valid_json(jsonb_array_elements_text(a.preguntas_y_aclaraciones)) AS condicion
        FROM soce_descripcion b
            JOIN soce_fechas sf ON b.contract_id::text = sf.contract_id::text
            JOIN preguntas_y_aclaraciones a ON b.contract_id::integer = a.contract_id
        WHERE jsonb_typeof(a.preguntas_y_aclaraciones) = 'array'::text 
            AND sf.fecha_name::text = 'fecha_de_publicacion'::text 
            AND b.tipo_de_contratacion::text = 'Empresas Públicas, Mercantiles o Subsidiarias'::text
    ) x
    WHERE x.condicion IS TRUE 
        AND (x.estado_del_proceso = ANY (ARRAY[
            'Adjudicada'::text, 'Cancelado'::text, 'Desierta'::text, 
            'Finalizada'::text, 'Por Adjudicar'::text, 
            'Preguntas, Respuestas y Aclaraciones'::text, 
            'Audiencia de Preguntas y Aclaraciones'::text, 
            'Terminado Unilateralmente'::text, 'En Curso'::text, 'Suspendido'::text
        ]))
) z
WHERE EXTRACT(YEAR FROM z.fecha_publicacion) = 2023
    AND z.valor_preguntas::text <> '"ACLARACION"'
ORDER BY z.fecha_publicacion, z.contract_id;

--SIE filtrado por aclaracion--

SELECT 
    z.contract_id,
    z.codigo,
    z.fecha_publicacion,
    z.estado_del_proceso,
    p.presupuesto_referencial_total_sin_iva,
    p.monto_contrato,
    p.monto_adjudicacion,
    z.valor_preguntas,
    z.flag AS es_acusatoria,
    CASE 
        WHEN z.flag THEN 'Acusatoria'
        ELSE 'No Acusatoria'
    END AS clasificacion
FROM (
    SELECT 
        x.contract_id,
        x.codigo,
        x.estado_del_proceso,
        x.fecha_publicacion,
        (SELECT jsonb_each.value -> 'pregunta_aclaracion'::text
           FROM jsonb_each(x.preguntas) jsonb_each(key, value)) AS valor_preguntas,
        to_tsvector('spanish'::regconfig, ((SELECT jsonb_each.value -> 'pregunta_aclaracion'::text
           FROM jsonb_each(x.preguntas) jsonb_each(key, value)))::text) 
            @@ to_tsquery('spanish'::regconfig, 'corrupción|direccionado|limitante|vulneración|ocultamiento|violación|incompleto|trato<->justo'::text) AS flag
    FROM (
        SELECT 
            b.contract_id,
            b.codigo,
            sf.fecha_value::date AS fecha_publicacion,
            CASE substring(b.estado_del_proceso::text, 1, 1)
                WHEN '{'::text THEN b.estado_del_proceso::text
                ELSE ('{"estado_del_proceso":"'::text || b.estado_del_proceso::text) || '"}'::text
            END::jsonb ->> 'estado_del_proceso'::text AS estado_del_proceso,
            jsonb_array_elements(a.preguntas_y_aclaraciones) AS preguntas,
            is_valid_json(jsonb_array_elements_text(a.preguntas_y_aclaraciones)) AS condicion
        FROM soce_descripcion b
            JOIN soce_fechas sf ON b.contract_id::text = sf.contract_id::text
            JOIN preguntas_y_aclaraciones a ON b.contract_id::integer = a.contract_id
        WHERE jsonb_typeof(a.preguntas_y_aclaraciones) = 'array'::text 
            AND b.tipo_de_contratacion::text = 'Subasta Inversa Electrónica'::text 
            AND sf.fecha_name::text = 'fecha_de_publicacion'::text
    ) x
    WHERE x.condicion IS TRUE 
        AND (x.estado_del_proceso = ANY (ARRAY[
            'Preguntas, Respuestas y Aclaraciones'::text, 'Entrega de Propuesta'::text,
            'Convalidacion de Errores'::text, 'Calificación de Participantes'::text,
            'Oferta Inicial'::text, 'Puja'::text, 'Negociación'::text,
            'Reprogramación Puja'::text, 'Suspendido'::text, 'Por Adjudicar'::text,
            'Adjudicado - Registro de Contratos'::text, 'Adjudicada'::text,
            'Ejecución de Contrato'::text, 'En Recepción'::text, 'Finalizada'::text,
            'Terminado Unilateralmente'::text, 'Finalizado por mutuo acuerdo'::text,
            'Finalizado por disolución de la persona Jurídica'::text,
            'Finalizado a solicitud del contratista'::text, 'Cancelado'::text, 'Desierta'::text
        ]))
) z
LEFT JOIN mv_sie_procesos_datos p ON z.contract_id::text = p.contract_id::text 
    AND z.codigo::text = p.codigo::text
WHERE EXTRACT(YEAR FROM z.fecha_publicacion) = 2024
    AND z.valor_preguntas::text <> '"ACLARACION"'
ORDER BY z.fecha_publicacion, z.contract_id;

--124584 2024--   ----121839 2025------

--consulta general--

SELECT 
    z.contract_id,
    z.codigo,
    z.fecha_publicacion,
    z.estado_del_proceso,
    p.presupuesto_referencial_total_sin_iva,
    p.monto_contrato,
    p.monto_adjudicacion,
    z.valor_preguntas,
    z.flag AS es_acusatoria,
    CASE 
        WHEN z.flag THEN 'Acusatoria'
        ELSE 'No Acusatoria'
    END AS clasificacion
FROM (
    SELECT 
        x.contract_id,
        x.codigo,
        x.estado_del_proceso,
        x.fecha_publicacion,
        (SELECT jsonb_each.value -> 'pregunta_aclaracion'::text
           FROM jsonb_each(x.preguntas) jsonb_each(key, value)) AS valor_preguntas,
        to_tsvector('spanish'::regconfig, ((SELECT jsonb_each.value -> 'pregunta_aclaracion'::text
           FROM jsonb_each(x.preguntas) jsonb_each(key, value)))::text)
            @@ to_tsquery('spanish'::regconfig, 'corrupción|direccionado|limitante|vulneración|ocultamiento|violación|incompleto|trato<->justo'::text) AS flag
    FROM (
        -- GEN: Publicación Especial
        SELECT DISTINCT
            b.contract_id,
            b.codigo,
            sf.fecha_value::date AS fecha_publicacion,
            CASE substring(b.estado_del_proceso::text, 1, 1)
                WHEN '{'::text THEN b.estado_del_proceso::text
                ELSE ('{"estado_del_proceso":"'::text || b.estado_del_proceso::text) || '"}'::text
            END::jsonb ->> 'estado_del_proceso'::text AS estado_del_proceso,
            jsonb_array_elements(a.preguntas_y_aclaraciones) AS preguntas,
            is_valid_json(jsonb_array_elements_text(a.preguntas_y_aclaraciones)) AS condicion
        FROM soce_descripcion b
            JOIN soce_fechas sf ON b.contract_id::text = sf.contract_id::text
            JOIN preguntas_y_aclaraciones a ON b.contract_id::integer = a.contract_id
        WHERE jsonb_typeof(a.preguntas_y_aclaraciones) = 'array'::text
            AND sf.fecha_name::text = 'fecha_de_publicacion'::text
            AND b.tipo_de_contratacion::text = 'Publicación Especial'::text
            AND valida_codigo_gen_pe(b.codigo)

        UNION

        -- GEN: Empresas Públicas, Mercantiles o Subsidiarias
        SELECT DISTINCT
            b.contract_id,
            b.codigo,
            sf.fecha_value::date AS fecha_publicacion,
            CASE substring(b.estado_del_proceso::text, 1, 1)
                WHEN '{'::text THEN b.estado_del_proceso::text
                ELSE ('{"estado_del_proceso":"'::text || b.estado_del_proceso::text) || '"}'::text
            END::jsonb ->> 'estado_del_proceso'::text AS estado_del_proceso,
            jsonb_array_elements(a.preguntas_y_aclaraciones) AS preguntas,
            is_valid_json(jsonb_array_elements_text(a.preguntas_y_aclaraciones)) AS condicion
        FROM soce_descripcion b
            JOIN soce_fechas sf ON b.contract_id::text = sf.contract_id::text
            JOIN preguntas_y_aclaraciones a ON b.contract_id::integer = a.contract_id
        WHERE jsonb_typeof(a.preguntas_y_aclaraciones) = 'array'::text
            AND sf.fecha_name::text = 'fecha_de_publicacion'::text
            AND b.tipo_de_contratacion::text = 'Empresas Públicas, Mercantiles o Subsidiarias'::text

        UNION ALL

        -- SIE: Subasta Inversa Electrónica
        SELECT
            b.contract_id,
            b.codigo,
            sf.fecha_value::date AS fecha_publicacion,
            CASE substring(b.estado_del_proceso::text, 1, 1)
                WHEN '{'::text THEN b.estado_del_proceso::text
                ELSE ('{"estado_del_proceso":"'::text || b.estado_del_proceso::text) || '"}'::text
            END::jsonb ->> 'estado_del_proceso'::text AS estado_del_proceso,
            jsonb_array_elements(a.preguntas_y_aclaraciones) AS preguntas,
            is_valid_json(jsonb_array_elements_text(a.preguntas_y_aclaraciones)) AS condicion
        FROM soce_descripcion b
            JOIN soce_fechas sf ON b.contract_id::text = sf.contract_id::text
            JOIN preguntas_y_aclaraciones a ON b.contract_id::integer = a.contract_id
        WHERE jsonb_typeof(a.preguntas_y_aclaraciones) = 'array'::text
            AND sf.fecha_name::text = 'fecha_de_publicacion'::text
            AND b.tipo_de_contratacion::text = 'Subasta Inversa Electrónica'::text
    ) x
    WHERE x.condicion IS TRUE
        AND (x.estado_del_proceso = ANY (ARRAY[
            -- Estados GEN
            'Adjudicada'::text, 'Cancelado'::text, 'Desierta'::text,
            'Finalizada'::text, 'Por Adjudicar'::text,
            'Preguntas, Respuestas y Aclaraciones'::text,
            'Audiencia de Preguntas y Aclaraciones'::text,
            'Terminado Unilateralmente'::text, 'En Curso'::text, 'Suspendido'::text,
            -- Estados exclusivos SIE
            'Entrega de Propuesta'::text, 'Convalidacion de Errores'::text,
            'Calificación de Participantes'::text, 'Oferta Inicial'::text,
            'Puja'::text, 'Negociación'::text, 'Reprogramación Puja'::text,
            'Adjudicado - Registro de Contratos'::text, 'Ejecución de Contrato'::text,
            'En Recepción'::text, 'Finalizado por mutuo acuerdo'::text,
            'Finalizado por disolución de la persona Jurídica'::text,
            'Finalizado a solicitud del contratista'::text
        ]))
) z
LEFT JOIN mv_sie_procesos_datos p ON z.contract_id::text = p.contract_id::text
    AND z.codigo::text = p.codigo::text
WHERE EXTRACT(YEAR FROM z.fecha_publicacion) = 2022
    AND z.valor_preguntas::text <> '"ACLARACION"'
ORDER BY z.fecha_publicacion, z.contract_id;



--consulta general preguntas ---

SELECT 
    z.valor_preguntas AS valor_preguntas,
    z.flag AS es_acusatoria,
    CASE 
        WHEN z.flag THEN 'Acusatoria'
        ELSE 'No Acusatoria'
    END AS clasificacion
FROM (
    SELECT 
        x.contract_id,
        x.codigo,
        x.estado_del_proceso,
        x.fecha_publicacion,
        (SELECT jsonb_each.value ->> 'pregunta_aclaracion'
           FROM jsonb_each(x.preguntas) jsonb_each(key, value)) AS valor_preguntas,
        to_tsvector('spanish'::regconfig, ((SELECT jsonb_each.value ->> 'pregunta_aclaracion'
           FROM jsonb_each(x.preguntas) jsonb_each(key, value)))::text)
            @@ to_tsquery('spanish'::regconfig, 'corrupción|direccionado|limitante|vulneración|ocultamiento|violación|incompleto|trato<->justo'::text) AS flag
    FROM (
        -- GEN: Publicación Especial
        SELECT DISTINCT
            b.contract_id,
            b.codigo,
            sf.fecha_value::date AS fecha_publicacion,
            CASE substring(b.estado_del_proceso::text, 1, 1)
                WHEN '{'::text THEN b.estado_del_proceso::text
                ELSE ('{"estado_del_proceso":"'::text || b.estado_del_proceso::text) || '"}'::text
            END::jsonb ->> 'estado_del_proceso'::text AS estado_del_proceso,
            jsonb_array_elements(a.preguntas_y_aclaraciones) AS preguntas,
            is_valid_json(jsonb_array_elements_text(a.preguntas_y_aclaraciones)) AS condicion
        FROM soce_descripcion b
            JOIN soce_fechas sf ON b.contract_id::text = sf.contract_id::text
            JOIN preguntas_y_aclaraciones a ON b.contract_id::integer = a.contract_id
        WHERE jsonb_typeof(a.preguntas_y_aclaraciones) = 'array'::text
            AND sf.fecha_name::text = 'fecha_de_publicacion'::text
            AND b.tipo_de_contratacion::text = 'Publicación Especial'::text
            AND valida_codigo_gen_pe(b.codigo)

        UNION

        -- GEN: Empresas Públicas, Mercantiles o Subsidiarias
        SELECT DISTINCT
            b.contract_id,
            b.codigo,
            sf.fecha_value::date AS fecha_publicacion,
            CASE substring(b.estado_del_proceso::text, 1, 1)
                WHEN '{'::text THEN b.estado_del_proceso::text
                ELSE ('{"estado_del_proceso":"'::text || b.estado_del_proceso::text) || '"}'::text
            END::jsonb ->> 'estado_del_proceso'::text AS estado_del_proceso,
            jsonb_array_elements(a.preguntas_y_aclaraciones) AS preguntas,
            is_valid_json(jsonb_array_elements_text(a.preguntas_y_aclaraciones)) AS condicion
        FROM soce_descripcion b
            JOIN soce_fechas sf ON b.contract_id::text = sf.contract_id::text
            JOIN preguntas_y_aclaraciones a ON b.contract_id::integer = a.contract_id
        WHERE jsonb_typeof(a.preguntas_y_aclaraciones) = 'array'::text
            AND sf.fecha_name::text = 'fecha_de_publicacion'::text
            AND b.tipo_de_contratacion::text = 'Empresas Públicas, Mercantiles o Subsidiarias'::text

        UNION ALL

        -- SIE: Subasta Inversa Electrónica
        SELECT
            b.contract_id,
            b.codigo,
            sf.fecha_value::date AS fecha_publicacion,
            CASE substring(b.estado_del_proceso::text, 1, 1)
                WHEN '{'::text THEN b.estado_del_proceso::text
                ELSE ('{"estado_del_proceso":"'::text || b.estado_del_proceso::text) || '"}'::text
            END::jsonb ->> 'estado_del_proceso'::text AS estado_del_proceso,
            jsonb_array_elements(a.preguntas_y_aclaraciones) AS preguntas,
            is_valid_json(jsonb_array_elements_text(a.preguntas_y_aclaraciones)) AS condicion
        FROM soce_descripcion b
            JOIN soce_fechas sf ON b.contract_id::text = sf.contract_id::text
            JOIN preguntas_y_aclaraciones a ON b.contract_id::integer = a.contract_id
        WHERE jsonb_typeof(a.preguntas_y_aclaraciones) = 'array'::text
            AND sf.fecha_name::text = 'fecha_de_publicacion'::text
            AND b.tipo_de_contratacion::text = 'Subasta Inversa Electrónica'::text
    ) x
    WHERE x.condicion IS TRUE
        AND (x.estado_del_proceso = ANY (ARRAY[
            'Adjudicada'::text, 'Cancelado'::text, 'Desierta'::text,
            'Finalizada'::text, 'Por Adjudicar'::text,
            'Preguntas, Respuestas y Aclaraciones'::text,
            'Audiencia de Preguntas y Aclaraciones'::text,
            'Terminado Unilateralmente'::text, 'En Curso'::text, 'Suspendido'::text,
            'Entrega de Propuesta'::text, 'Convalidacion de Errores'::text,
            'Calificación de Participantes'::text, 'Oferta Inicial'::text,
            'Puja'::text, 'Negociación'::text, 'Reprogramación Puja'::text,
            'Adjudicado - Registro de Contratos'::text, 'Ejecución de Contrato'::text,
            'En Recepción'::text, 'Finalizado por mutuo acuerdo'::text,
            'Finalizado por disolución de la persona Jurídica'::text,
            'Finalizado a solicitud del contratista'::text
        ]))
) z
WHERE EXTRACT(YEAR FROM z.fecha_publicacion) = 2022
    AND z.valor_preguntas <> 'ACLARACION'
    AND z.valor_preguntas IS NOT NULL
ORDER BY z.fecha_publicacion, z.contract_id;









--consulta de historic data--

Select * from historic_data limit 100;

select count(*) from preguntas_y_aclaraciones;  

select count(*) from soce_universo;


SELECT 
    z.contract_id,
    z.codigo,
    z.fecha_publicacion,
    z.estado_del_proceso,
    z.valor_preguntas,
    z.flag AS es_acusatoria,
    CASE 
        WHEN z.flag THEN 'Acusatoria'
        ELSE 'No Acusatoria'
    END AS clasificacion
FROM (
    SELECT 
        x.contract_id,
        x.codigo,
        x.estado_del_proceso,
        x.fecha_publicacion,
        (SELECT jsonb_each.value -> 'pregunta_aclaracion'::text
           FROM jsonb_each(x.preguntas) jsonb_each(key, value)) AS valor_preguntas,
        to_tsvector('spanish'::regconfig, ((SELECT jsonb_each.value -> 'pregunta_aclaracion'::text
           FROM jsonb_each(x.preguntas) jsonb_each(key, value)))::text) 
            @@ to_tsquery('spanish'::regconfig, 'corrupción|direccionado|limitante|vulneración|ocultamiento|violación|incompleto|trato<->justo'::text) AS flag
    FROM (
        SELECT DISTINCT 
            b.contract_id,
            b.codigo,
            sf.fecha_value::date AS fecha_publicacion,
            CASE substring(b.estado_del_proceso::text, 1, 1)
                WHEN '{'::text THEN b.estado_del_proceso::text
                ELSE ('{"estado_del_proceso":"'::text || b.estado_del_proceso::text) || '"}'::text
            END::jsonb ->> 'estado_del_proceso'::text AS estado_del_proceso,
            jsonb_array_elements(a.preguntas_y_aclaraciones) AS preguntas,
            is_valid_json(jsonb_array_elements_text(a.preguntas_y_aclaraciones)) AS condicion
        FROM soce_descripcion b
            JOIN soce_fechas sf ON b.contract_id::text = sf.contract_id::text
            JOIN preguntas_y_aclaraciones a ON b.contract_id::integer = a.contract_id
        WHERE jsonb_typeof(a.preguntas_y_aclaraciones) = 'array'::text 
            AND sf.fecha_name::text = 'fecha_de_publicacion'::text 
            AND b.tipo_de_contratacion::text = 'Publicación Especial'::text 
            AND valida_codigo_gen_pe(b.codigo)
        UNION
        SELECT DISTINCT 
            b.contract_id,
            b.codigo,
            sf.fecha_value::date AS fecha_publicacion,
            CASE "substring"(b.estado_del_proceso::text, 1, 1)
                WHEN '{'::text THEN b.estado_del_proceso::text
                ELSE ('{"estado_del_proceso":"'::text || b.estado_del_proceso::text) || '"}'::text
            END::jsonb ->> 'estado_del_proceso'::text AS estado_del_proceso,
            jsonb_array_elements(a.preguntas_y_aclaraciones) AS preguntas,
            is_valid_json(jsonb_array_elements_text(a.preguntas_y_aclaraciones)) AS condicion
        FROM soce_descripcion b
            JOIN soce_fechas sf ON b.contract_id::text = sf.contract_id::text
            JOIN preguntas_y_aclaraciones a ON b.contract_id::integer = a.contract_id
        WHERE jsonb_typeof(a.preguntas_y_aclaraciones) = 'array'::text 
            AND sf.fecha_name::text = 'fecha_de_publicacion'::text 
            AND b.tipo_de_contratacion::text = 'Empresas Públicas, Mercantiles o Subsidiarias'::text
    ) x
    WHERE x.condicion IS TRUE 
        AND (x.estado_del_proceso = ANY (ARRAY[
            'Adjudicada'::text, 'Cancelado'::text, 'Desierta'::text, 
            'Finalizada'::text, 'Por Adjudicar'::text, 
            'Preguntas, Respuestas y Aclaraciones'::text, 
            'Audiencia de Preguntas y Aclaraciones'::text, 
            'Terminado Unilateralmente'::text, 'En Curso'::text, 'Suspendido'::text
        ]))
) z
WHERE EXTRACT(YEAR FROM z.fecha_publicacion) = 2023
    AND z.valor_preguntas::text <> '"ACLARACION"'
ORDER BY z.fecha_publicacion, z.contract_id;


---------------------------pruebas con historic data--------------------------------------

SELECT 
    z.contract_id,
    z.codigo,
    z.fecha_publicacion,
    z.estado_del_proceso,
    z.valor_preguntas,
    z.flag AS es_acusatoria,
    CASE 
        WHEN z.flag THEN 'Acusatoria'
        ELSE 'No Acusatoria'
    END AS clasificacion
FROM (
    SELECT 
        x.contract_id,
        x.codigo,
        x.estado_del_proceso,
        x.fecha_publicacion,
        (SELECT jsonb_each.value -> 'pregunta_aclaracion'
           FROM jsonb_each(x.pregunta) jsonb_each(key, value)) AS valor_preguntas,
        to_tsvector('spanish'::regconfig, 
            ((SELECT jsonb_each.value -> 'pregunta_aclaracion'
                FROM jsonb_each(x.pregunta) jsonb_each(key, value)))::text
        ) @@ to_tsquery('spanish'::regconfig, 
            'corrupción|direccionado|limitante|vulneración|ocultamiento|violación|incompleto|trato<->justo'
        ) AS flag
    FROM (
        SELECT DISTINCT
            hd.contract_id,
            hd.data_json -> 'descripcion' ->> 'codigo'                  AS codigo,
            (                                                             -- fecha_de_publicacion desde el array de fechas
                SELECT f.value ->> 'fecha_value'
                FROM jsonb_array_elements(hd.data_json -> 'fechas') AS f(value)
                WHERE f.value ->> 'fecha_name' = 'fecha_de_publicacion'
                LIMIT 1
            )::date                                                       AS fecha_publicacion,
            hd.data_json -> 'descripcion' ->> 'estado_del_proceso'       AS estado_del_proceso,
            jsonb_array_elements(hd.data_json -> 'preguntas_y_aclaraciones') AS pregunta
        FROM historic_data hd
        WHERE
            -- solo registros donde preguntas_y_aclaraciones es un array JSON
            jsonb_typeof(hd.data_json -> 'preguntas_y_aclaraciones') = 'array'
            AND hd.data_json -> 'descripcion' ->> 'tipo_de_contratacion' 
                IN ('Publicación Especial', 'Empresas Públicas, Mercantiles o Subsidiarias')
            AND valida_codigo_gen_pe(hd.data_json -> 'descripcion' ->> 'codigo')
    ) x
    WHERE 
        x.estado_del_proceso = ANY (ARRAY[
            'Adjudicada', 'Cancelado', 'Desierta',
            'Finalizada', 'Por Adjudicar',
            'Preguntas, Respuestas y Aclaraciones',
            'Audiencia de Preguntas y Aclaraciones',
            'Terminado Unilateralmente', 'En Curso', 'Suspendido'
        ])
) z
WHERE 
    EXTRACT(YEAR FROM z.fecha_publicacion) = 2022
    AND z.valor_preguntas::text <> '"ACLARACION"'
ORDER BY z.fecha_publicacion, z.contract_id;

