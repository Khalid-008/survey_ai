
class SurveyResult:
    def __init__(
        self,
        id: str = "",
        subject: str = "",
        description: str = "",
        question: str = "",
        answer: str = "",
    ):
        self.id = id
        self.subject = subject
        self.description = description
        self.question = question
        self.answer = answer