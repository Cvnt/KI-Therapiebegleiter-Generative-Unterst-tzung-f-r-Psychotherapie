# KI-Therapiebegleiter-Generative-Unterst-tzung-f-r-Psychotherapie
KI-Therapiebegleiter nutzt generative KI und Sprachmodelle, um Therapeuten bei der Vorbereitung von Sitzungen, der Analyse von Patientenaussagen und der Erstellung von personalisierten Behandlungsplänen zu unterstützen.
from transformers import pipeline, GPT2LMHeadModel, GPT2Tokenizer

# Initialize the model and tokenizer for generative tasks
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Initialize sentiment analysis pipeline
sentiment_pipeline = pipeline("sentiment-analysis")

def generate_session_topics(patient_history, goals):
    """Generate session topics based on patient's history and goals."""
    prompt = f"Based on the following patient history: {patient_history} and their goals: {goals}, generate session topics:"
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    output = model.generate(input_ids, max_length=100, num_return_sequences=1)
    return tokenizer.decode(output[0], skip_special_tokens=True)

def analyze_patient_statements(statements):
    """Analyze sentiment and key themes in patient statements."""
    results = sentiment_pipeline(statements)
    return results

def create_treatment_plan(patient_history, goals, session_analysis):
    """Generate a personalized treatment plan."""
    prompt = f"Create a personalized treatment plan based on patient history: {patient_history}, their goals: {goals}, and session analysis: {session_analysis}"
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    output = model.generate(input_ids, max_length=200, num_return_sequences=1)
    return tokenizer.decode(output[0], skip_special_tokens=True)

# Example Usage
patient_history = "The patient has been experiencing anxiety and stress related to work."
goals = "Patient aims to reduce anxiety and improve work-life balance."
session_analysis = "Patient expresses a strong negative sentiment towards work but positive feelings when discussing hobbies and leisure activities."

# Generate Session Topics
session_topics = generate_session_topics(patient_history, goals)
print("Session Topics:", session_topics)

# Analyze Patient Statements
patient_statements = "I just feel overwhelmed by everything at work. I can't find time for myself."
analysis_results = analyze_patient_statements([patient_statements])
print("Analysis of Patient Statements:", analysis_results)

# Create Treatment Plan
treatment_plan = create_treatment_plan(patient_history, goals, str(analysis_results))
print("Treatment Plan:", treatment_plan)
