# utils.py

import numpy as np
import os
import logging
from groq import Groq
import re # For fallback parsing
# Optional: Import for similarity check if needed later
# import difflib

# Import transformers and handle potential errors
try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logging.warning("Transformers library not installed. EQ scoring will use fallback. Install with 'pip install transformers torch' or 'pip install transformers tensorflow'")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedEQAssessmentModel:
    def __init__(self):
        # Initialize Groq client (for polishing, validation, question generation)
        self.groq_client = None
        try:
            if 'GROQ_API_KEY' in os.environ:
                self.groq_client = Groq(api_key=os.environ.get('GROQ_API_KEY'))
            else:
                logger.warning("GROQ_API_KEY not found in environment variables. Using provided fallback key for Groq features.")
                # Using the provided fallback key - Essential for question generation now
                self.groq_client = Groq(api_key="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") # Replace if needed
        except Exception as e:
            logger.error(f"Error initializing Groq client: {e}")
            # If Groq fails to initialize, question generation will fail later.

        # Initialize Sentiment Analysis Pipeline (for EQ scoring)
        self.sentiment_analyzer = None
        if TRANSFORMERS_AVAILABLE:
            try:
                # Using the same model as the Streamlit example
                self.sentiment_analyzer = pipeline(
                    'sentiment-analysis',
                    model='distilbert-base-uncased-finetuned-sst-2-english'
                )
                logger.info("Sentiment analysis pipeline initialized successfully.")
            except Exception as e:
                logger.error(f"Error initializing sentiment analysis pipeline: {e}. EQ scoring will use fallback logic.")
                self.sentiment_analyzer = None # Ensure it's None on failure
        else:
            logger.warning("Transformers library not available. EQ scoring will use fallback logic.")


        # Scoring categories
        self.scoring_categories = [
            'Self-Awareness',
            'Self-Regulation',
            'Emotion Management',
            'Emotional Resilience',
            'Supportiveness',
            'Motivation',
            'Trust and Relationship Building',
            'Cultural Awareness',
            'Situational Awareness'
        ]
        # Max score per category will now be 20, based on Streamlit's logic
        self.max_category_score = 20.0


    # --- Functions using Groq (Polishing, Validation, Question Generation) ---

    def polish_response(self, question_text, raw_response_text):
        """Polish the raw response text using Groq API for grammar, deduplication, and contextual correction, with minimal changes."""
        if not self.groq_client:
            logger.warning("Groq client not available for polishing, returning raw response.")
            return raw_response_text

        # Trim whitespace just in case
        raw_response_text = raw_response_text.strip()

        if not raw_response_text or len(raw_response_text) < 3: # Don't polish very short or empty responses
            logger.info("Raw response too short for polishing, returning raw response as is.")
            return raw_response_text

        try:
            # --- Start of Updated Prompt ---
            prompt_content = (
                "You are an AI assistant performing EXTREMELY MINIMAL edits on user responses to interview questions. "
                "Your ONLY task is to improve basic linguistic clarity by making the absolute minimum necessary changes. "
                "Your PRIMARY DIRECTIVE: **ABSOLUTELY PRESERVE the user's original meaning, sentiment, specific information, and intent, NO MATTER WHAT.** Even if the response is rude, irrelevant, nonsensical, or uncooperative, you MUST reflect that accurately after minimal cleaning. The polished response length must be very similar to the raw response length.\n\n"

                "Perform ONLY these actions on the 'Raw Response' (using 'Question' only for context on potential word misrecognition):\n"
                "1.  Correct obvious grammatical errors (tense, agreement, articles, plurals, basic prepositions). Example: 'He go store' -> 'He goes to the store'.\n"
                "2.  Remove clear stuttering (e.g., 'I-I-I am' -> 'I am') and excessive word/phrase repetitions (e.g., 'and and then' -> 'and then'). Do not remove intentional repetition for emphasis if not excessive.\n"
                "3.  Correct clearly misrecognized words ONLY if the context from the 'Question' makes the correction highly certain. Example: Question about 'deadline', Raw Response 'dead lion is tomorrow' -> correct to 'deadline is tomorrow'.\n\n"

                "**CRITICAL PROHIBITIONS (Strictly FORBIDDEN actions):**\n"
                "-   **DO NOT change the core meaning or intent.** If the user says 'I don't want to talk to you', the polished response MUST reflect refusal, like 'I don't want to talk to you.' It MUST NOT become something like 'I'm not sure how to answer that.' or 'I need more time.'\n"
                "-   **DO NOT add ANY new information, ideas, reasons, examples, or actions.**\n"
                "-   **DO NOT elaborate, expand, or rephrase sentences significantly.** Minor reordering for basic clarity is permissible ONLY if meaning is identical.\n"
                "-   **DO NOT change the user's expressed sentiment or tone** (e.g., don't make a negative response neutral).\n"
                "-   **DO NOT change pronouns or references** unless grammatically incorrect.\n"
                "-   The output length must remain very close to the input length.\n\n"

                "**Example of handling irrelevant/uncooperative input:**\n"
                "Question: 'How would you handle a team conflict?'\n"
                "Raw Response: 'uhh i i like pizza its good'\n"
                "Correct Polished Response: 'Uh, I like pizza. It's good.' (Only corrected 'uhh', 'i i', 'its', and added punctuation.)\n"
                "Incorrect Polished Response (DO NOT DO THIS): 'I'm not sure how I'd handle conflict, maybe we could discuss it over pizza?' (This adds ideas and changes intent).\n\n"

                "Output ONLY the polished response text. No explanations, no apologies, no labels like 'Polished Response:'. Just the text.\n\n"

                f"Question: \"{question_text}\"\n"
                f"Raw Response: \"{raw_response_text}\"\n\n"
                "Polished Response:"
            )
            # --- End of Updated Prompt ---

            chat_completion = self.groq_client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are a highly constrained text editor. Your sole task is to apply minimal grammatical corrections and cleanups (stuttering, obvious misrecognition) to the user's raw response. You MUST preserve the original meaning, sentiment, and information EXACTLY, even if the response is off-topic or uncooperative. Do NOT add, rephrase significantly, or 'improve' the answer. Adhere strictly to the user's prompt."
                    },
                    {
                        "role": "user",
                        "content": prompt_content
                    }
                ],
                model="llama3-8b-8192",
                temperature=0.05, # Even lower temperature for maximum fidelity
                max_tokens=int(len(raw_response_text.split()) * 1.5 + 30) # Slightly adjusted max_tokens, tighter multiplier
            )

            polished_text = chat_completion.choices[0].message.content.strip()

            # --- Post-Polishing Checks (Optional but recommended for robustness) ---
            # Heuristic check: If the polished text is drastically different, revert.
            # This requires a similarity metric. Using a simple length check for now.
            raw_len = len(raw_response_text)
            pol_len = len(polished_text)

            if not polished_text:
                logger.warning(f"Polished text is empty. Raw: '{raw_response_text}'. Reverting to raw response.")
                return raw_response_text

            # More aggressive length check for potential over-editing or hallucination
            # Allow for some shrinkage due to stutter removal, but not drastic expansion/contraction.
            if raw_len > 15: # Apply stricter checks only on slightly longer inputs
                 # Allow shrinkage (e.g. stutter removal) down to 1/3rd, but expansion limited to 2x
                if pol_len < raw_len / 3 or pol_len > raw_len * 2.0:
                    logger.warning(f"Polished text length significantly different from raw. Raw ({raw_len} chars): '{raw_response_text}', Polished ({pol_len} chars): '{polished_text}'. Reverting to raw response as precaution.")
                    return raw_response_text

            # Optional: More advanced check using sequence similarity (e.g., difflib)
            # try:
            #     # Calculate similarity ratio
            #     similarity = difflib.SequenceMatcher(None, raw_response_text.lower(), polished_text.lower()).ratio()
            #     # Define a threshold (e.g., 0.6 - adjust based on testing)
            #     SIMILARITY_THRESHOLD = 0.6
            #     if similarity < SIMILARITY_THRESHOLD:
            #         logger.warning(f"Polished text similarity ({similarity:.2f}) below threshold ({SIMILARITY_THRESHOLD}). Raw: '{raw_response_text}', Polished: '{polished_text}'. Reverting to raw response.")
            #         return raw_response_text
            # except Exception as sim_err:
            #     logger.error(f"Error during similarity check: {sim_err}")
            # --- End Post-Polishing Checks ---

            logger.info(f"Response polished. Raw: '{raw_response_text}'. Polished: '{polished_text}'")
            return polished_text

        except Exception as e:
            logger.error(f"Error polishing response: {e}. Returning raw response.")
            return raw_response_text

    # --- Rest of the class methods (validate_answer, generate_questions, etc.) remain unchanged ---
    def validate_answer(self, question, answer):
        """Validate that the answer is appropriate for the question."""
        if not answer or len(answer.strip()) < 10:
            logger.info(f"Answer too short post-polishing or empty: '{answer}'")
            return False

        try:
            if not self.groq_client:
                logger.warning("Groq client not available for validation, using length check.")
                return len(answer.strip()) >= 20 # Adjusted stricter length if no LLM

            chat_completion = self.groq_client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You need to assess if the given 'Answer' is a valid attempt to respond to the 'Question'. \n"
                            "Validation Criteria:\n"
                            "1. Substantive: Is the answer more than just a few words (e.g., at least 3 meaningful words)?\n"
                            "2. Language: Should be in English. Minor grammar mistakes are acceptable.\n"
                            "3. Relevance: Does the answer attempt to address the question, even if the content of the answer itself is poor or simple? The validation should be liberal: an answer is valid if it seems like an attempt to answer the question, however flawed. An answer is INVALID only if it's completely off-topic (e.g., 'I like pizza' to a question about workplace conflict), gibberish, or an explicit refusal to answer (like 'Leave me alone').\n" # Added explicit refusal to INVALID criteria
                            f"Question: {question}\n\n"
                            f"Answer: {answer}\n\n"
                            "Respond ONLY with 'VALID' or 'INVALID'. Do not explain or add any additional text."
                        )
                    }
                ],
                model="llama3-8b-8192",
                max_tokens=10
            )

            validation_response = chat_completion.choices[0].message.content.strip().upper()

            logger.info(f"Validation API response for Q: '{question}' A: '{answer}' -> '{validation_response}'")
            if validation_response == 'VALID':
                return True
            elif validation_response == 'INVALID':
                return False
            else:
                logger.warning(f"Validation response was unclear: '{validation_response}'. Applying manual length check as fallback.")
                # Make fallback check slightly stricter if validation failed/unclear
                return len(answer.strip()) >= 25

        except Exception as e:
            logger.error(f"Error in answer validation: {e}")
            # Make fallback check slightly stricter if validation failed
            return len(answer.strip()) >= 25

    # --- generate_questions method ---
    def generate_questions(self, age, profession, name):
        """
        Generate personalized EQ questions using Groq AI.
        MUST use AI-generated questions. Returns empty list on failure.
        """
        # Check if Groq client is available. If not, cannot generate questions.
        if not self.groq_client:
            logger.error("Groq client is not available. Cannot generate questions.")
            return [] # Return empty list indicating failure

        try:
            chat_completion = self.groq_client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            f"You are an AI specializing in creating Emotional Intelligence (EQ) assessment questions. "
                            f"Your task is to generate exactly 10 unique, open-ended EQ scenarios/questions. "
                            f"Crucially, EACH of these 10 questions must be specifically tailored for a {age}-year-old individual named {name} working as a {profession}. "
                            f"The questions must reflect realistic professional challenges, interpersonal dynamics, or ethical dilemmas relevant to the '{profession}' field. "
                            f"Avoid generic EQ questions; every question must be contextualized for their role as a {profession}. "
                            "The goal is to elicit responses that reveal emotional intelligence within their specific professional environment. "
                            "Provide a variety of scenarios. \n\n"

                            "VERY IMPORTANT INSTRUCTION ON QUESTION STYLE:\n"
                            "1.  ABSOLUTELY AVOID questions that ask the user to recall specific past events. DO NOT use phrasing like 'Describe a time when...', 'Tell me about a time...', 'Can you share an example of a time...', 'Recall a situation where...', or any similar retrospective questions.\n"
                            "2.  INSTEAD, ALL questions MUST be framed as PRESENT or FUTURE HYPOTHETICAL scenarios. The aim is to assess their problem-solving approach, reasoning, and emotional handling in a given hypothetical context, not their ability to recount past experiences.\n"
                            "3.  USE PHRASING LIKE: 'Imagine you are in a situation where...', 'How would you handle a scenario where...', 'What would be your approach if...', 'Consider that [X happens], how would you respond as a {profession}?', 'Suppose [Y situation occurs], what steps would you take?', 'If faced with [Z challenge], what would be your primary concerns and actions?'\n\n"

                            "Formatting Instructions:\n"
                            "- Each question must start on a new line.\n"
                            "- The first question (Q1) and the tenth question (Q10) must start *only* with 'Q: ' followed by the question text (e.g., 'Q: Imagine your team is facing...').\n"
                            "- Questions 2 through 9 (Q2 to Q9) must start with a brief, polite acknowledgment of the previous answer, then 'Q: ', then the question text (e.g., 'Thanks for that perspective. Q: Now, suppose your project lead...').\n"
                            "- Output ONLY the 10 questions formatted this way. No other text, preambles, or explanations.\n"
                            f"Example of a GOOD profession-specific question (for a 'Software Engineer'): 'Q: As a Software Engineer, imagine you discover a significant flaw in a colleague's code just before a critical release. How would you approach your colleague about this, especially if they are known to be defensive?'\n"
                            f"Example of a BAD question (retrospective and generic - TO AVOID): 'Q: Describe a time you dealt with a defensive colleague.'\n"
                            f"Note that if the profession is student,then don't use the words colleague, manager, team lead in the questions but use classmates, mentor, professor etc..."
                        )
                    },
                    {
                        "role": "user",
                        "content": f"Please generate the 10 EQ questions for {name} ({profession}, {age} years old). Ensure all are specific to the {profession} role and STRICTLY AVOID any questions asking to 'describe a time when' or recall past events. Focus entirely on hypothetical scenarios using phrases like 'Imagine if...' or 'How would you handle...'."
                    }
                ],
                model="llama3-8b-8192",
                temperature=0.7,
                max_tokens=1800
            )

            response_text = chat_completion.choices[0].message.content
            logger.info(f"Raw response from Groq for questions: {response_text[:300]}...") # Log beginning of response

            questions = []
            raw_lines = response_text.split('\n')
            for line in raw_lines:
                line = line.strip()
                if not line:
                    continue

                # Robust parsing to find the question part after potential pleasantries and "Q:"
                q_marker_index = line.rfind("Q:")
                if q_marker_index != -1:
                    question_text = line[q_marker_index + len("Q:"):].strip()
                    if question_text and question_text.endswith('?'): # Basic check for question format
                        questions.append(question_text)
                # Fallback parsing (less reliable) - attempt if primary fails
                elif not questions and len(line) > 20 and line.endswith('?') and not line.startswith("Thanks"):
                     logger.debug(f"Trying fallback parsing for line: {line}")
                     questions.append(line) # Less ideal, might capture non-questions

            # Attempt fallback re-parsing if primary parsing yields no results but response text exists
            if not questions and response_text.strip():
                 logger.warning("Primary 'Q:' marker parsing failed. Attempting fallback newline split parsing.")
                 temp_questions = []
                 for line_fb in response_text.split('\n'):
                     line_fb = line_fb.strip()
                     if not line_fb: continue
                     # Remove common prefixes
                     line_fb = re.sub(r"^(Q:\s*|\d+\.\s*|Thanks for [^.]+\.\s*Q:\s*)", "", line_fb, flags=re.IGNORECASE).strip()
                     if len(line_fb) > 15 and line_fb.endswith('?'):
                         temp_questions.append(line_fb)
                 if temp_questions:
                     questions = temp_questions
                     logger.info(f"Fallback parsing collected {len(questions)} potential questions.")
                 else:
                    logger.error(f"Fallback parsing also failed to extract questions from response: {response_text}")


            logger.info(f"Successfully parsed {len(questions)} questions. First few: {questions[:3] if questions else 'None'}.")

            # --- Strict check: Ensure we have enough questions ---
            # We require exactly 10 questions as per the prompt. Adjust if flexibility is needed.
            MIN_REQUIRED_QUESTIONS = 10
            if len(questions) < MIN_REQUIRED_QUESTIONS:
                logger.error(f"Failed to generate the required number of questions ({len(questions)} generated, need {MIN_REQUIRED_QUESTIONS}). Returning empty list.")
                return [] # Indicate failure

            # Return exactly 10 questions if more were somehow generated
            return questions[:MIN_REQUIRED_QUESTIONS]

        except Exception as e:
            logger.error(f"An unexpected error occurred during question generation: {e}", exc_info=True)
            # Do NOT fall back to default questions. Return empty list to signal failure.
            return []

    # --- EQ Scoring based on Sentiment Analysis ---
    def _analyze_sentiment_for_scoring(self, response_text):
        """
        Analyze response using sentiment analysis pipeline.
        Mirrors Streamlit's analyze_emotional_response logic.
        Returns a dictionary with sentiment label and score.
        """
        # Fallback if pipeline isn't available or response is too short
        if not self.sentiment_analyzer or not response_text or len(response_text.strip()) < 5:
            logger.warning(f"Sentiment analyzer not available or response too short ('{response_text}'). Returning neutral sentiment.")
            return {
                'sentiment_label': 'NEUTRAL',
                'sentiment_score': 0.5, # Neutral score equivalent
                'response_length': len(response_text or "")
            }

        try:
            # Run sentiment analysis
            results = self.sentiment_analyzer(response_text)
            if not results: # Handle empty result list
                 logger.warning(f"Sentiment analysis returned empty result for: '{response_text}'. Using neutral.")
                 return {'sentiment_label': 'NEUTRAL', 'sentiment_score': 0.5, 'response_length': len(response_text)}

            sentiment_result = results[0]
            label = sentiment_result.get('label', 'NEUTRAL').upper() # Ensure uppercase standard
            score = sentiment_result.get('score', 0.5) # Default to neutral score if missing

            logger.info(f"Sentiment analysis for response: '{response_text[:50]}...' -> Label: {label}, Score: {score:.4f}")

            # Translate Streamlit's 'emotions' dict concept roughly using sentiment
            emotions = {
                'positive': score * 10 if label == 'POSITIVE' else 0,
                'negative': score * 10 if label == 'NEGATIVE' else 0,
                'neutral': 5 if label == 'NEUTRAL' else 0 # Give neutral a base score as in Streamlit
            }

            return {
                'sentiment_label': label,
                'sentiment_score': score,
                'response_length': len(response_text),
                'emotions': emotions # Pass the derived 'emotions' dict
            }

        except Exception as e:
            logger.error(f"Error during sentiment analysis for response '{response_text[:50]}...': {e}. Returning neutral.")
            return {
                'sentiment_label': 'NEUTRAL',
                'sentiment_score': 0.5,
                'response_length': len(response_text or ""),
                'emotions': {'neutral': 5} # Fallback emotions
            }

    # --- create_eq_assessment method ---
    def create_eq_assessment(self, age, profession, gender, name, responses):
        """
        Create an EQ assessment based on user responses using sentiment analysis.
        Mirrors Streamlit's create_eq_assessment logic.
        """
        # Perform sentiment analysis on each response
        response_analyses = [self._analyze_sentiment_for_scoring(resp) for resp in responses]

        # Initialize base scores (using Streamlit's starting value of 10.0)
        base_scores = {cat: 10.0 for cat in self.scoring_categories}

        # Simplified scoring mechanism based on Streamlit's logic
        for analysis in response_analyses:
            sentiment_label = analysis.get('sentiment_label', 'NEUTRAL')
            emotions = analysis.get('emotions', {})

            if sentiment_label == 'POSITIVE':
                base_scores['Emotional Resilience'] += 1.0
                base_scores['Motivation'] += 0.8
            elif sentiment_label == 'NEGATIVE':
                base_scores['Self-Regulation'] -= 0.6
                base_scores['Emotional Resilience'] -= 0.4

            if 'positive' in emotions and emotions['positive'] > 0:
                 base_scores['Supportiveness'] += emotions['positive'] * 0.03
                 base_scores['Self-Awareness'] += emotions['positive'] * 0.02
            if 'negative' in emotions and emotions['negative'] > 0:
                 base_scores['Trust and Relationship Building'] -= emotions['negative'] * 0.015
                 base_scores['Self-Regulation'] -= emotions['negative'] * 0.01
                 # Note: Streamlit logic had Trust and Relationship Building decreased twice by negative emotions, keeping it here
                 base_scores['Trust and Relationship Building'] -= emotions['negative'] * 0.01

        # Gender adjustments (from Streamlit logic)
        gender_lower = gender.lower() if gender else ""
        if gender_lower == 'male':
            base_scores['Self-Regulation'] += 0.5
            base_scores['Trust and Relationship Building'] += 0.3
        elif gender_lower == 'female':
            base_scores['Supportiveness'] += 1.0
            base_scores['Emotional Resilience'] += 0.8
        elif gender_lower == 'non-binary':
            base_scores['Cultural Awareness'] += 1.2
            base_scores['Trust and Relationship Building'] += 0.5
        # Note: No explicit 'other' or default adjustment in original Streamlit logic

        # Normalize scores and calculate total EQ (clamping between 0 and 20 per category)
        normalized_scores = {}
        total_eq = 0.0
        for category in self.scoring_categories:
            score = base_scores.get(category, 10.0) # Get score, default 10
            normalized_score = max(0.0, min(self.max_category_score, score)) # Clamp between 0 and max
            normalized_scores[category] = round(normalized_score, 2) # Round for consistency
            total_eq += normalized_scores[category] # Sum rounded scores

        # Round the overall EQ score as well
        total_eq = round(total_eq, 2)

        return {
            'overall_eq': total_eq,
            'category_scores': normalized_scores, # Already rounded
            'response_analyses': [{
                'sentiment_label': r.get('sentiment_label'),
                'sentiment_score': round(r.get('sentiment_score', 0.5), 4),
                'response_length': r.get('response_length')
             } for r in response_analyses]
        }


    # --- Interpretation Logic ---
    def interpret_eq_score(self, eq_score):
        """Interpret the EQ score based on Streamlit's ranges."""
        # Max possible score = 9 categories * 20 points/category = 180
        # Ranges based on Streamlit logic's implied total max ~160-180
        if eq_score < 70:
            return "Low EQ", "Current emotional intelligence level indicates significant opportunities for development"
        elif 70 <= eq_score < 80:
            return "Below Average EQ", "Shows basic emotional awareness with room for growth"
        elif 80 <= eq_score < 100:
            return "Average EQ", "Demonstrates typical emotional intelligence with good foundation"
        elif 100 <= eq_score < 120:
            return "Good EQ", "Shows strong emotional intelligence capabilities"
        elif 120 <= eq_score < 140:
            return "High EQ", "Exhibits advanced emotional intelligence skills"
        elif 140 <= eq_score < 160:
            return "Very High EQ", "Demonstrates exceptional emotional intelligence" # Adjusted label slightly
        else: # eq_score >= 160
            return "Extraordinarily High EQ", "Shows remarkable mastery of emotional intelligence"


    # --- Improvement Suggestions ---
    def generate_improvement_suggestions(self, category_scores):
        """Generate personalized improvement suggestions based on the weakest categories."""
        suggestions = {
            'Self-Awareness': [
                "Practice daily reflection: Spend 5-10 minutes reviewing your day, noting emotions and triggers.",
                "Keep an emotion journal: Write down situations and how they made you feel, without judgment.",
                "Seek feedback: Ask trusted friends or colleagues for honest input on your emotional responses and behaviors.",
                "Mindfulness meditation: Engage in short mindfulness exercises to increase present-moment awareness."
            ],
            'Self-Regulation': [
                "Identify your triggers: Note situations or words that cause strong emotional reactions.",
                "Practice the 'pause': Before reacting in a heated moment, take a deep breath and count to ten.",
                "Develop coping mechanisms: Find healthy ways to manage stress, like exercise, hobbies, or talking it out.",
                "Reframe negative thoughts: Challenge and try to replace negative self-talk with more balanced perspectives."
            ],
            'Emotion Management': [
                "Name your emotions: Accurately identify what you're feeling (e.g., 'frustrated' instead of just 'bad').",
                "Accept your emotions: Understand that all emotions are valid, even uncomfortable ones. Focus on how you respond to them.",
                "Practice emotional expression: Find appropriate ways to express your feelings, rather than bottling them up or exploding.",
                "Use stress-reduction techniques: Explore deep breathing, progressive muscle relaxation, or guided imagery."
            ],
            'Emotional Resilience': [
                "Build a strong support network: Cultivate relationships with positive, supportive people.",
                "Focus on what you can control: Let go of things outside your influence.",
                "View setbacks as learning opportunities: Analyze what went wrong and what can be done differently.",
                "Practice self-compassion: Treat yourself with kindness and understanding during difficult times."
            ],
            'Supportiveness': [
                "Practice active listening: Focus fully on what others are saying, both verbally and non-verbally.",
                "Show empathy: Try to understand and share the feelings of others from their perspective.",
                "Offer help and encouragement: Be proactive in supporting colleagues and friends.",
                "Validate others' feelings: Acknowledge their emotions even if you don't agree with their viewpoint (e.g., 'I can see why you'd feel that way')."
            ],
            'Motivation': [
                "Set clear, meaningful goals: Align your tasks with your values and larger aspirations.",
                "Break down large goals: Divide big objectives into smaller, manageable steps to build momentum.",
                "Celebrate small wins: Acknowledge and reward your progress along the way.",
                "Maintain a positive outlook: Focus on possibilities and solutions, even when faced with challenges."
            ],
            'Trust and Relationship Building': [
                "Be reliable and consistent: Follow through on your commitments.",
                "Communicate openly and honestly (with tact): Share your thoughts and feelings appropriately.",
                "Show vulnerability appropriately: Sharing struggles can build deeper connections.",
                "Resolve conflicts constructively: Address disagreements directly and respectfully, seeking win-win solutions."
            ],
            'Cultural Awareness': [
                "Educate yourself: Learn about different cultures, traditions, and perspectives.",
                "Be curious and respectful: Ask open-minded questions and listen without judgment.",
                "Challenge your own biases: Be aware of stereotypes and actively work to overcome them.",
                "Adapt your communication style: Be mindful of cultural nuances in language and behavior."
            ],
            'Situational Awareness': [
                "Observe non-verbal cues: Pay attention to body language, tone of voice, and facial expressions.",
                "Read the room: Assess the emotional atmosphere of a group or situation.",
                "Consider context: Understand how the environment and circumstances influence interactions.",
                "Anticipate reactions: Think about how your words and actions might affect others before you act."
            ]
        }

        sorted_categories = sorted(category_scores.items(), key=lambda x: x[1])
        lowest_categories = sorted_categories[:3] # Get top 3 lowest scoring categories

        improvement_plan = {}
        for category, score in lowest_categories:
            # Only suggest improvements if score is below a threshold (e.g., less than 15/20)
            if score < 15.0 and category in suggestions:
                improvement_plan[category] = {
                    'score': score, # Already rounded
                    'suggestions': suggestions[category],
                    # Provide specific actionable items
                    'easy_start': suggestions[category][0] if suggestions[category] else "Reflect on this area.",
                    'daily_practice': suggestions[category][1] if len(suggestions[category]) > 1 else "Continue exploring this area."
                }
            # Limit to max 3 suggestions even if more are below threshold
            if len(improvement_plan) >= 3:
                break

        return improvement_plan


# --- Chart Data Generation ---
def generate_chart_data(category_scores):
    """Generate chart data for visualization, using MAX SCORE = 20."""
    max_score_per_category = 20.0

    categories = list(category_scores.keys())
    # Ensure scores used are the already rounded and clamped ones from category_scores dict
    scores = [category_scores.get(cat, 0.0) for cat in categories]

    chart_data = []
    for i, category in enumerate(categories):
        score_val = scores[i]
        percentage = round((score_val / max_score_per_category) * 100, 1) if max_score_per_category > 0 else 0
        chart_data.append({
            'category': category.replace(' ', '\n'), # Add newline for better display in chart labels
            'score': score_val, # Use the rounded score
            'percentage': percentage
        })

    return chart_data