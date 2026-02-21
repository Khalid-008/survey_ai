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

def get_survey_df(survey_id):
    try:
        conn = get_conn()
        result = pd.read_sql(f"""SELECT 
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
            WHERE s.survey_number = '{survey_id}'
            """, conn)
        
        print(f"DEBUG: get_survey_df({survey_id}) returned {len(result)} rows")
        return result
    finally:
        conn.close()


def get_selection_questions_data(survey_number: str) -> pd.DataFrame:
    """
    تجلب جميع الأسئلة غير النصية (غير TEXT_INPUT) مع إجاباتها
    المرتبطة بالاستبيان المحدد بـ survey_number.

    تستخدم COALESCE(sa.answer, sa.selected_options_id) لتغطية:
    - أسئلة الخيار الفردي  (answer)
    - أسئلة الاختيار المتعدد (قد تُخزَّن في selected_options_id)

    العمود الناتج:
        question_id     : معرّف السؤال
        question_ar     : نص السؤال بالعربية
        question_type   : نوع السؤال
        answer          : إجابة المستجيب (من answer أو selected_options_id)
        submission_id   : معرّف الاستجابة الواحدة
    """
    query = """
        SELECT
            sq.id               AS question_id,
            sq.question_ar      AS question_ar,
            sq.question_type    AS question_type,
            COALESCE(
                NULLIF(TRIM(sa.answer), ''),
                NULLIF(TRIM(sa.selected_options_id), '')
            )                   AS answer,
            sa.submission_id    AS submission_id
        FROM ms_survey_service.survey_question sq
        INNER JOIN ms_survey_service.survey_answer sa
            ON sq.id = sa.survey_question_id
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
    """
    conn = get_conn()
    try:
        df = pd.read_sql(query, conn, params=(survey_number,))
        if not df.empty:
            types = df["question_type"].unique().tolist()
            print(f"DEBUG: get_selection_questions_data({survey_number}) → "
                  f"{len(df)} rows | types: {types}")
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