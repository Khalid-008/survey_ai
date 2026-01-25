from data.models import SurveyResult
from data.connection import conn
import pandas as pd
import warnings

# Suppress pandas SQLAlchemy warning for DBAPI2 connections
warnings.filterwarnings('ignore', message='.*SQLAlchemy connectable.*')



def get_survey_questions(survey_id):
    query = """
    SELECT q.ID as QuestionID, q.Question as Questions, q.Question_Type as QuestionType, a.Answer
    FROM [VMS].[dbo].[Survey] s
    INNER JOIN SurveyQuestion q ON q.SurveyID = s.ID
    INNER JOIN SurveyAnswer a ON q.ID = a.SQID
    WHERE s.ID = ?
    """
    df = pd.read_sql(query, conn, params=[survey_id])
    return df.to_json()

def run_query(query: str):
    try:
        df = pd.read_sql(query, conn)
        
        print(f"✅ Query executed successfully, returned {len(df)} rows")
        return df.to_json(force_ascii=False)
    except Exception as e:
        print(f"❌ Query execution failed!")
        print(f"Error: {str(e)}")
        print(f"Query: {query}")
        raise

def get_distinct_answers_for_questions(question_ids: list[str]) -> dict:
    """
    Get distinct answers for a list of question IDs.
    Returns a dictionary mapping question_id -> list of distinct answers
    """
    if not question_ids:
        return {}
    
    # Build the query to get distinct answers for all questions
    placeholders = ', '.join(['?' for _ in question_ids])
    query = f"""
    SELECT 
        T1.ID as QuestionID,
        T1.Question as QuestionText,
        T2.Answer
    FROM SurveyQuestion AS T1
    JOIN SurveyAnswer AS T2 ON T1.ID = T2.SQID
    WHERE T1.ID IN ({placeholders})
    GROUP BY T1.ID, T1.Question, T2.Answer
    ORDER BY T1.ID, T2.Answer
    """
    
    df = pd.read_sql(query, conn, params=question_ids)
    
    # Group by question ID and collect distinct answers
    result = {}
    for question_id in question_ids:
        question_df = df[df['QuestionID'] == question_id]
        if not question_df.empty:
            result[question_id] = {
                'question_text': question_df.iloc[0]['QuestionText'],
                'distinct_answers': question_df['Answer'].tolist()
            }
    
    return result