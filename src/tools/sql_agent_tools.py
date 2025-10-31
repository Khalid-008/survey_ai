from data.operations import get_survey_questions

# def convert_question_to_query(survey_id : str ,question : str) -> str:

#     survey_id = "6BB381D8-B348-4E66-8CCA-A9F63A91973F"
#     question = "how many result do I have"
#     # test 
#     result = get_survey_questions(survey_id)

#     # 2 based on the question should the agent get the needed question to make the insight 

#     query = ""

#     return query 

def get_survey_questions(survey_id: str) -> dict:
    """
    Retrieves all questions for a given survey ID.

    Args:
        survey_id (str): The unique identifier of the survey.

    Returns:
        dict: A dictionary containing the survey questions.
    """
    questions_json = get_survey_questions(survey_id)
    import json
    try:
        return json.loads(questions_json)
    except Exception:
        return {"error": "Failed to parse survey questions."}

