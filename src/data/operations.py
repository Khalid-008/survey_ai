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
                q.id as QuestionID, 
                q.question_ar as Questions, 
                q.question_type as QuestionType, 
                a.answer as Answer
            FROM ms_survey_service.survey s
            INNER JOIN ms_survey_service.survey_question q ON q.survey_id = s.id
            INNER JOIN ms_survey_service.survey_answer a ON q.id = a.survey_question_id
            WHERE s.survey_number = '{survey_id}'""", conn)
        
        print(f"DEBUG: get_survey_df({survey_id}) returned {len(result)} rows")
        return result
    finally:
        conn.close()