from data.models import SurveyResult
from data.connection import conn
import pandas as pd



def get_survey_questions(survey_id):
    query = """
    SELECT q.ID as QuestionID, q.Question as Questions, q.Question_Type as QuestionType
    FROM [VMS].[dbo].[Survey] s
    INNER JOIN SurveyQuestion q ON q.SurveyID = s.ID
    WHERE s.ID = ?
    """
    df = pd.read_sql(query, conn, params=[survey_id])
    return df.to_json()

def run_query(query: str):
    df = pd.read_sql(query, conn)
    return df.to_json(force_ascii=False)
