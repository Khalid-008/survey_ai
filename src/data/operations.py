import pandas as pd
import warnings
import mysql.connector
import os
from dotenv import load_dotenv

load_dotenv()

def get_conn():
    return mysql.connector.connect(
            host=os.getenv("DB_HOST"),
            port=int(os.getenv("DB_PORT", 3306)),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            database=os.getenv("DB_DATABASE"),
            connection_timeout=60,
            buffered=True
        )

def get_survey_df(survey_id, date_from=None, date_to=None):
    try:
        conn = get_conn()

        date_filter = ""
        params = [survey_id]
        if date_from:
            date_filter += " AND a.created_date >= %s"
            params.append(date_from)
        if date_to:
            date_filter += " AND a.created_date <= %s"
            params.append(date_to)

        query = f"""SELECT 
                a.external_id as ExternalID,
                s.survey_number as SurveyNumber,
                s.subject as SurveyTitle,
                q.id as QuestionID, 
                q.question_ar as Questions, 
                q.question_type as QuestionType, 
                a.answer as Answer
            FROM ms_survey_service.survey s
            INNER JOIN ms_survey_service.survey_question q ON q.survey_id = s.id
            INNER JOIN ms_survey_service.survey_answer a ON q.id = a.survey_question_id
            WHERE s.survey_number = %s
            {date_filter}
            """
        result = pd.read_sql(query, conn, params=params)
        
        print(f"DEBUG: get_survey_df({survey_id}) returned {len(result)} rows"
              + (f" [filter: {date_from} → {date_to}]" if date_from or date_to else ""))
        return result
    finally:
        conn.close()


def get_selection_questions_data(survey_number: str, date_from=None, date_to=None) -> pd.DataFrame:

    date_filter = ""
    params = [survey_number]
    if date_from:
        date_filter += " AND sa.created_date >= %s"
        params.append(date_from)
    if date_to:
        date_filter += " AND sa.created_date <= %s"
        params.append(date_to)

    query = f"""
        SELECT
            sq.id               AS question_id,
            sq.question_ar      AS question_ar,
            sq.question_type    AS question_type,
            COALESCE(
                NULLIF(TRIM(sa.answer), ''),
                NULLIF(TRIM(sa.selected_options_id), '')
            )                   AS answer,
            sa.submission_id    AS submission_id,
            sa.created_date     AS submission_date,
            CASE
                WHEN UPPER(sq.question_type) IN ('MULTIPLE_CHOICE', 'DROPDOWN')
                    THEN qo.option_text_ar

                WHEN UPPER(sq.question_type) = 'YES_NO'
                    THEN CASE COALESCE(NULLIF(TRIM(sa.answer),''), NULLIF(TRIM(sa.selected_options_id),''))
                            WHEN '1' THEN 'نعم'
                            WHEN '0' THEN 'لا'
                            ELSE COALESCE(NULLIF(TRIM(sa.answer),''), NULLIF(TRIM(sa.selected_options_id),''))
                         END

                WHEN UPPER(sq.question_type) = 'EMOJIS'
                    THEN CASE COALESCE(NULLIF(TRIM(sa.answer),''), NULLIF(TRIM(sa.selected_options_id),''))
                            WHEN '1' THEN 'Very Dissatisfied'
                            WHEN '2' THEN 'Dissatisfied'
                            WHEN '3' THEN 'Neutral'
                            WHEN '4' THEN 'Satisfied'
                            WHEN '5' THEN 'Very Satisfied'
                            ELSE COALESCE(NULLIF(TRIM(sa.answer),''), NULLIF(TRIM(sa.selected_options_id),''))
                         END

                ELSE COALESCE(
                        NULLIF(TRIM(sa.answer), ''),
                        NULLIF(TRIM(sa.selected_options_id), '')
                     )
            END                 AS answer_label
        FROM ms_survey_service.survey_question sq
        INNER JOIN ms_survey_service.survey_answer sa
            ON sq.id = sa.survey_question_id
        LEFT JOIN ms_survey_service.question_option qo
            ON  qo.question_id = sq.id
            AND qo.id = CAST(
                    COALESCE(
                        NULLIF(TRIM(sa.answer), ''),
                        NULLIF(TRIM(sa.selected_options_id), '')
                    ) AS UNSIGNED
                )
        WHERE sq.survey_id = (
            SELECT id FROM ms_survey_service.survey
            WHERE survey_number = %s
            LIMIT 1
        )
        AND UPPER(sq.question_type) != 'TEXT_INPUT'
        AND COALESCE(
            NULLIF(TRIM(sa.answer), ''),
            NULLIF(TRIM(sa.selected_options_id), '')
        ) IS NOT NULL
        {date_filter}
    """
    conn = get_conn()
    try:
        df = pd.read_sql(query, conn, params=params)
        if not df.empty:
            types = df["question_type"].unique().tolist()
            print(f"DEBUG: get_selection_questions_data({survey_number}) → "
                  f"{len(df)} rows | types: {types}"
                  + (f" [filter: {date_from} → {date_to}]" if date_from or date_to else ""))
        else:
            print(f"DEBUG: get_selection_questions_data({survey_number}) → 0 rows")
        return df
    finally:
        conn.close()



def execute_raw_query(query: str) -> pd.DataFrame:
    """
    تنفّذ كويري MySQL خام وتُرجع النتائج كـ DataFrame.
    تُستخدم لتشغيل الكويريات التي يولّدها الـ LLM.
    """
    conn = get_conn()
    try:
        df = pd.read_sql(query, conn)
        print(f"DEBUG: execute_raw_query() returned {len(df)} rows, {len(df.columns)} columns")
        return df
    finally:
        conn.close()