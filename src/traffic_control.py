from model import model_it
from score import score_it

class Traffic_Control(model_it):
    def __init__(self):
        self.model_name = None

    def _model_it(self, model_name):
        model_it(model_name)

    def score_it(self, model_name):
        score_it(model_name)

    def first_stage(self):
        model_it('first')
        score_it('first')

    def second_stage(self):
        create
