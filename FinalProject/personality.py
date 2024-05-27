class Personality:
    def __init__(self, openness, conscientiousness, extraversion, agreeableness, neuroticism):
        self.openness = openness
        self.conscientiousness = conscientiousness
        self.extraversion = extraversion
        self.agreeableness = agreeableness
        self.neuroticism = neuroticism

    def __repr__(self):
        return f"Personality(O={self.openness}, C={self.conscientiousness}, E={self.extraversion}, A={self.agreeableness}, N={self.neuroticism})"
